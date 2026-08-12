import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from pathlib import Path
from datetime import datetime
from scipy.stats import poisson, norm
import warnings
import time

warnings.filterwarnings('ignore')

# ================= 全局参数配置 =================
# ========== 数据相关参数 ==========
WINDOW_SIZE = 128          # 滑动窗口大小，使用过去多少期数据预测下一期
NUM_NUMBERS = 33          # 红球数字范围（1-33）
RECENT_WINDOW = 3479      # 计算近期频率特征的窗口大小
TARGET_COLS = ['Red1', 'Red2', 'Red3', 'Red4', 'Red5', 'Red6']  # 目标列名，即红球6个位置

# ========== 训练相关参数 ==========
BATCH_SIZE = 128           # 批次大小，每批处理的样本数，增大可提高GPU/CPU利用率
EPOCHS = 128              # 最大训练轮数
LEARNING_RATE = 0.001     # 初始学习率，控制参数更新步长

# ========== 模型结构参数 ==========
LSTM_HIDDEN_SIZE = 64     # LSTM隐藏层神经元数量，越大模型容量越大但训练越慢
LSTM_NUM_LAYERS = 2       # LSTM层数，多层可捕捉更复杂模式但训练更慢
DROPOUT_RATE = 0.1        # Dropout率，防止过拟合，0表示不使用dropout
FC_HIDDEN_SIZE = 32       # 全连接层神经元数量

# ========== 早停与学习率调度参数 ==========
EARLY_STOP_PATIENCE = 7   # 早停耐心值，验证集损失连续多少轮不下降则停止训练
LR_SCHEDULER_FACTOR = 0.5 # 学习率衰减因子，验证集损失不下降时乘以该值
LR_SCHEDULER_PATIENCE = 3 # 学习率调度耐心值，连续多少轮不下降则衰减学习率
VAL_FREQUENCY = 10        # 验证频率，每多少个epoch进行一次验证

# ========== 数据划分参数 ==========
TEST_SIZE = 0.3           # 测试集比例，从总数据中划分多少作为测试集
VAL_SIZE = 0.5            # 验证集比例，从测试集中划分多少作为验证集（剩余为测试集）


# ================= 特征工程模块（向量化优化）=================
def calc_odd_even_features(df):
    """计算每个号码的奇偶标志特征（向量化）"""
    for i, col in enumerate(TARGET_COLS):
        df[f'Odd_Even_{i+1}'] = (df[col] % 2).astype(np.float32)
    return df


def calc_recent_frequency(df):
    """计算最近RECENT_WINDOW期内每个数字的出现频率（向量化）"""
    red_data = df[TARGET_COLS].values
    
    for num in range(1, NUM_NUMBERS + 1):
        mask = (red_data == num).astype(np.float32).sum(axis=1)
        df[f'Recent_Freq_{num}'] = pd.Series(mask).rolling(window=RECENT_WINDOW, min_periods=1).sum()
    
    return df


def calc_last_appearance(df):
    """计算每个数字距离上次出现的期数（向量化）"""
    red_data = df[TARGET_COLS].values
    
    for num in range(1, NUM_NUMBERS + 1):
        mask = (red_data == num).astype(np.float32).sum(axis=1)
        positions = np.where(mask == 1)[0]
        
        if len(positions) == 0:
            df[f'Last_Appear_{num}'] = np.arange(1, len(df) + 1, dtype=np.float32)
        else:
            expanded = np.zeros(len(df), dtype=np.float32)
            expanded[positions] = positions.astype(np.float32) + 1
            last_pos = np.maximum.accumulate(expanded)
            last_pos[last_pos == 0] = np.nan
            result = np.arange(1, len(df) + 1, dtype=np.float32) - last_pos
            df[f'Last_Appear_{num}'] = np.nan_to_num(result, nan=np.arange(1, len(df) + 1, dtype=np.float32))
    
    return df


def calc_poisson_features(df, window_size=100):
    """计算每个号码在滑动窗口内的泊松分布概率（向量化）"""
    red_data = df[TARGET_COLS].values
    
    for num in range(1, NUM_NUMBERS + 1):
        mask = (red_data == num).astype(np.float32).sum(axis=1)
        freq_col = pd.Series(mask).rolling(window=window_size, min_periods=1).sum()
        lam = freq_col / window_size
        df[f'Poisson_{num}'] = poisson.pmf(1, lam)
    
    return df


def calc_normal_features(df):
    """计算红球和值的正态分布参数特征（向量化）"""
    df['Sum'] = df[TARGET_COLS].sum(axis=1)
    df['Sum_Mean'] = df['Sum'].rolling(window=50, min_periods=10).mean().fillna(df['Sum'].mean())
    df['Sum_Std'] = df['Sum'].rolling(window=50, min_periods=10).std().fillna(df['Sum'].std())
    df['Sum_Skew'] = df['Sum'].rolling(window=50, min_periods=10).skew().fillna(0)
    df['Sum_Kurt'] = df['Sum'].rolling(window=50, min_periods=10).kurt().fillna(0)
    
    return df


def calc_entropy_features(df, window_size=50):
    """计算号码分布的信息熵特征（优化版）"""
    red_data = df[TARGET_COLS].values
    n = len(df)
    entropy = np.zeros(n, dtype=np.float32)
    
    for i in range(window_size, n):
        window = red_data[i - window_size:i].flatten()
        _, counts = np.unique(window, return_counts=True)
        probs = counts / len(window)
        entropy[i] = -np.sum(probs * np.log2(probs + 1e-10))
    
    df['Entropy'] = entropy
    return df


def calc_markov_features(df):
    """计算奇偶比的马尔可夫链转移概率特征（优化版）"""
    df['Odd_Count'] = (df[TARGET_COLS] % 2 == 1).sum(axis=1).astype(np.int32)
    
    states = 7
    blue_odd_vals = df['Odd_Count'].values
    
    transition_matrix = np.zeros((states, states), dtype=np.float32)
    for i in range(1, len(df)):
        transition_matrix[blue_odd_vals[i - 1], blue_odd_vals[i]] += 1
    
    with np.errstate(divide='ignore', invalid='ignore'):
        transition_prob = transition_matrix / transition_matrix.sum(axis=1, keepdims=True)
        transition_prob = np.nan_to_num(transition_prob)
    
    for state in range(states):
        df[f'Markov_Prob_{state}'] = transition_prob[blue_odd_vals, state]
    
    return df


def calc_position_features(df):
    """计算每个位置号码的统计特征（向量化）"""
    for i, col in enumerate(TARGET_COLS):
        df[f'Pos_{i+1}_Mean'] = df[col].rolling(window=30, min_periods=5).mean().fillna(df[col].mean())
        df[f'Pos_{i+1}_Std'] = df[col].rolling(window=30, min_periods=5).std().fillna(df[col].std())
        df[f'Pos_{i+1}_Recent'] = df[col].rolling(window=3, min_periods=1).mean().fillna(df[col].mean())
    
    return df


def calc_consecutive_features(df):
    """计算连号特征：连号对数、最长连号长度、连号比例（向量化）"""
    red_data = df[TARGET_COLS].values
    n = len(df)
    
    consecutive_pairs = np.zeros(n, dtype=np.float32)
    max_consecutive = np.zeros(n, dtype=np.float32)
    
    for i in range(n):
        nums = np.sort(red_data[i])
        diffs = np.diff(nums)
        pairs = np.sum(diffs == 1)
        consecutive_pairs[i] = pairs
        
        max_len = 1
        current_len = 1
        for d in diffs:
            if d == 1:
                current_len += 1
                max_len = max(max_len, current_len)
            else:
                current_len = 1
        max_consecutive[i] = max_len
    
    df['Consecutive_Pairs'] = consecutive_pairs
    df['Max_Consecutive'] = max_consecutive
    df['Consecutive_Ratio'] = consecutive_pairs / 5.0
    
    df['Consecutive_Pairs_Mean'] = df['Consecutive_Pairs'].rolling(window=30, min_periods=5).mean().fillna(df['Consecutive_Pairs'].mean())
    df['Max_Consecutive_Mean'] = df['Max_Consecutive'].rolling(window=30, min_periods=5).mean().fillna(df['Max_Consecutive'].mean())
    
    return df


def calc_sum_interval_features(df):
    """计算和值区间特征：和值所在区间、各区间频率（向量化）"""
    if 'Sum' not in df.columns:
        df['Sum'] = df[TARGET_COLS].sum(axis=1)
    
    intervals = [(30, 50), (51, 70), (71, 90), (91, 110), (111, 130), (131, 150)]
    interval_labels = ['Sum_Int_1', 'Sum_Int_2', 'Sum_Int_3', 'Sum_Int_4', 'Sum_Int_5', 'Sum_Int_6']
    
    for label, (low, high) in zip(interval_labels, intervals):
        df[label] = ((df['Sum'] >= low) & (df['Sum'] <= high)).astype(np.float32)
    
    for label in interval_labels:
        df[f'{label}_Freq'] = df[label].rolling(window=50, min_periods=5).mean().fillna(df[label].mean())
    
    return df


def calc_hot_cold_features(df):
    """计算冷热号特征：近期热门号码数、冷门号码数、冷热比（向量化）"""
    red_data = df[TARGET_COLS].values
    
    freq_matrix = np.zeros((len(df), NUM_NUMBERS), dtype=np.float32)
    for num in range(1, NUM_NUMBERS + 1):
        freq_matrix[:, num - 1] = (red_data == num).astype(np.float32).sum(axis=1)
    
    recent_freq = np.zeros_like(freq_matrix)
    for i in range(len(df)):
        window_start = max(0, i - RECENT_WINDOW + 1)
        recent_freq[i] = freq_matrix[window_start:i + 1].sum(axis=0)
    
    hot_threshold = np.percentile(recent_freq, 70, axis=1)
    cold_threshold = np.percentile(recent_freq, 30, axis=1)
    
    hot_count = np.zeros(len(df), dtype=np.float32)
    cold_count = np.zeros(len(df), dtype=np.float32)
    
    for i in range(len(df)):
        nums = red_data[i]
        for num in nums:
            idx = int(num) - 1
            if recent_freq[i, idx] >= hot_threshold[i]:
                hot_count[i] += 1
            elif recent_freq[i, idx] <= cold_threshold[i]:
                cold_count[i] += 1
    
    df['Hot_Count'] = hot_count
    df['Cold_Count'] = cold_count
    df['Hot_Cold_Ratio'] = np.where(cold_count == 0, hot_count, hot_count / cold_count)
    
    df['Hot_Count_Mean'] = df['Hot_Count'].rolling(window=30, min_periods=5).mean().fillna(df['Hot_Count'].mean())
    df['Cold_Count_Mean'] = df['Cold_Count'].rolling(window=30, min_periods=5).mean().fillna(df['Cold_Count'].mean())
    
    return df


def calc_interval_distribution_features(df):
    """计算区间分布特征：各区间号码数量（向量化）"""
    red_data = df[TARGET_COLS].values
    
    intervals = [(1, 11), (12, 22), (23, 33)]
    interval_labels = ['Int_Dist_1', 'Int_Dist_2', 'Int_Dist_3']
    
    for label, (low, high) in zip(interval_labels, intervals):
        df[label] = np.sum((red_data >= low) & (red_data <= high), axis=1).astype(np.float32)
    
    df['Int_Dist_Max'] = df[interval_labels].max(axis=1)
    df['Int_Dist_Min'] = df[interval_labels].min(axis=1)
    df['Int_Dist_Std'] = df[interval_labels].std(axis=1)
    
    for label in interval_labels:
        df[f'{label}_Mean'] = df[label].rolling(window=30, min_periods=5).mean().fillna(df[label].mean())
    
    return df


def calc_prime_features(df):
    """计算质数特征：质数数量、质数比例（向量化）"""
    primes = {2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31}
    
    red_data = df[TARGET_COLS].values
    prime_mask = np.isin(red_data, list(primes)).astype(np.float32)
    
    df['Prime_Count'] = prime_mask.sum(axis=1)
    df['Prime_Ratio'] = df['Prime_Count'] / 6.0
    
    df['Prime_Count_Mean'] = df['Prime_Count'].rolling(window=30, min_periods=5).mean().fillna(df['Prime_Count'].mean())
    df['Prime_Ratio_Mean'] = df['Prime_Ratio'].rolling(window=30, min_periods=5).mean().fillna(df['Prime_Ratio'].mean())
    
    return df


def calc_size_distribution_features(df):
    """计算大小分布特征：大号数量、小号数量、大小比（向量化）"""
    red_data = df[TARGET_COLS].values
    
    df['Big_Count'] = np.sum(red_data > 16, axis=1).astype(np.float32)
    df['Small_Count'] = np.sum(red_data <= 16, axis=1).astype(np.float32)
    df['Big_Small_Ratio'] = np.where(df['Small_Count'] == 0, df['Big_Count'], df['Big_Count'] / df['Small_Count'])
    
    df['Big_Count_Mean'] = df['Big_Count'].rolling(window=30, min_periods=5).mean().fillna(df['Big_Count'].mean())
    df['Small_Count_Mean'] = df['Small_Count'].rolling(window=30, min_periods=5).mean().fillna(df['Small_Count'].mean())
    
    return df


def calc_sin_cos_features(df):
    """计算号码的正弦余弦编码特征（多周期配对编码，向量化）"""
    red_data = df[TARGET_COLS].values
    
    periods = [NUM_NUMBERS, 11, 3]
    
    for period in periods:
        for i, col in enumerate(TARGET_COLS):
            df[f'Sin_{i+1}_P{period}'] = np.sin(2 * np.pi * df[col] / period).astype(np.float32)
            df[f'Cos_{i+1}_P{period}'] = np.cos(2 * np.pi * df[col] / period).astype(np.float32)
    
    return df


def calc_law_of_large_numbers_features(df):
    """计算大数定律特征：每个号码的累计出现次数与理论期望的偏差（向量化）"""
    red_data = df[TARGET_COLS].values
    n = len(df)
    
    for num in range(1, NUM_NUMBERS + 1):
        mask = (red_data == num).astype(np.float32).sum(axis=1)
        cumulative_count = np.cumsum(mask)
        cumulative_expected = np.arange(1, n + 1, dtype=np.float32) / NUM_NUMBERS
        df[f'LLN_Deviation_{num}'] = cumulative_count - cumulative_expected
    
    df['LLN_Max_Deviation'] = df[[f'LLN_Deviation_{num}' for num in range(1, NUM_NUMBERS + 1)]].max(axis=1)
    df['LLN_Min_Deviation'] = df[[f'LLN_Deviation_{num}' for num in range(1, NUM_NUMBERS + 1)]].min(axis=1)
    df['LLN_Abs_Deviation_Mean'] = df[[f'LLN_Deviation_{num}' for num in range(1, NUM_NUMBERS + 1)]].abs().mean(axis=1)
    
    return df


def calculate_all_features(df):
    """计算所有特征的主入口函数（带时间统计）"""
    start_time = time.time()
    feature_times = {}
    
    print("\n" + "=" * 60)
    print("开始特征工程...")
    
    # 奇偶标志特征
    t_start = time.time()
    df = calc_odd_even_features(df)
    feature_times['奇偶标志'] = time.time() - t_start
    print(f"  [耗时 {feature_times['奇偶标志']:.2f}s] 奇偶标志特征")
    
    # 近期频率特征
    t_start = time.time()
    df = calc_recent_frequency(df)
    feature_times['近期频率'] = time.time() - t_start
    print(f"  [耗时 {feature_times['近期频率']:.2f}s] 近期频率特征")
    
    # 上次出现位置特征
    t_start = time.time()
    df = calc_last_appearance(df)
    feature_times['上次出现'] = time.time() - t_start
    print(f"  [耗时 {feature_times['上次出现']:.2f}s] 上次出现位置特征")
    
    # 泊松分布特征
    t_start = time.time()
    df = calc_poisson_features(df)
    feature_times['泊松分布'] = time.time() - t_start
    print(f"  [耗时 {feature_times['泊松分布']:.2f}s] 泊松分布特征")
    
    # 正态分布特征
    t_start = time.time()
    df = calc_normal_features(df)
    feature_times['正态分布'] = time.time() - t_start
    print(f"  [耗时 {feature_times['正态分布']:.2f}s] 正态分布特征")
    
    # 信息熵特征
    t_start = time.time()
    df = calc_entropy_features(df)
    feature_times['信息熵'] = time.time() - t_start
    print(f"  [耗时 {feature_times['信息熵']:.2f}s] 信息熵特征")
    
    # 马尔可夫链特征
    t_start = time.time()
    df = calc_markov_features(df)
    feature_times['马尔可夫链'] = time.time() - t_start
    print(f"  [耗时 {feature_times['马尔可夫链']:.2f}s] 马尔可夫链特征")
    
    # 位置统计特征
    t_start = time.time()
    df = calc_position_features(df)
    feature_times['位置统计'] = time.time() - t_start
    print(f"  [耗时 {feature_times['位置统计']:.2f}s] 位置统计特征")
    
    # 连号特征
    t_start = time.time()
    df = calc_consecutive_features(df)
    feature_times['连号特征'] = time.time() - t_start
    print(f"  [耗时 {feature_times['连号特征']:.2f}s] 连号特征")
    
    # 和值区间特征
    t_start = time.time()
    df = calc_sum_interval_features(df)
    feature_times['和值区间'] = time.time() - t_start
    print(f"  [耗时 {feature_times['和值区间']:.2f}s] 和值区间特征")
    
    # 冷热号特征
    t_start = time.time()
    df = calc_hot_cold_features(df)
    feature_times['冷热号'] = time.time() - t_start
    print(f"  [耗时 {feature_times['冷热号']:.2f}s] 冷热号特征")
    
    # 区间分布特征
    t_start = time.time()
    df = calc_interval_distribution_features(df)
    feature_times['区间分布'] = time.time() - t_start
    print(f"  [耗时 {feature_times['区间分布']:.2f}s] 区间分布特征")
    
    # 质数特征
    t_start = time.time()
    df = calc_prime_features(df)
    feature_times['质数特征'] = time.time() - t_start
    print(f"  [耗时 {feature_times['质数特征']:.2f}s] 质数特征")
    
    # 大小分布特征
    t_start = time.time()
    df = calc_size_distribution_features(df)
    feature_times['大小分布'] = time.time() - t_start
    print(f"  [耗时 {feature_times['大小分布']:.2f}s] 大小分布特征")
    
    # 正弦余弦编码特征
    t_start = time.time()
    df = calc_sin_cos_features(df)
    feature_times['正弦余弦'] = time.time() - t_start
    print(f"  [耗时 {feature_times['正弦余弦']:.2f}s] 正弦余弦编码特征")
    
    # 大数定律特征
    t_start = time.time()
    df = calc_law_of_large_numbers_features(df)
    feature_times['大数定律'] = time.time() - t_start
    print(f"  [耗时 {feature_times['大数定律']:.2f}s] 大数定律特征")
    
    total_time = time.time() - start_time
    print(f"\n特征工程总耗时: {total_time:.2f}秒")
    print(f"有效记录数: {len(df)}")
    
    return df.dropna().reset_index(drop=True)


# ================= 数据准备模块（优化版）=================
def extract_feature_columns(df):
    """提取所有特征列"""
    exclude_cols = ['dDate', 'dNum', 'yNum', 'mNum', 'Blue1', 'Sum', 'Odd_Count'] + TARGET_COLS
    feature_cols = [col for col in df.columns if col not in exclude_cols]
    return feature_cols


def create_sliding_windows(df, feature_cols, window_size):
    """创建滑动窗口数据集（向量化优化）"""
    start_time = time.time()
    
    n = len(df)
    X_data = df[feature_cols].values
    
    X = np.zeros((n - window_size, window_size, len(feature_cols)), dtype=np.float32)
    y = np.zeros((n - window_size, NUM_NUMBERS), dtype=np.float32)
    
    for i in range(n - window_size):
        X[i] = X_data[i:i + window_size]
        next_draw = df[TARGET_COLS].iloc[i + window_size].values
        for num in next_draw:
            y[i, int(num) - 1] = 1.0
    
    elapsed = time.time() - start_time
    print(f"滑动窗口生成完成 - X: {X.shape}, y: {y.shape}, 耗时: {elapsed:.2f}秒")
    
    return X, y


# ================= 数据集与模型模块 =================
class LSTMDataset(Dataset):
    """LSTM数据集类"""
    
    def __init__(self, X, y):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.float32)
    
    def __len__(self):
        return len(self.X)
    
    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


class ProbabilityLSTM(nn.Module):
    """LSTM概率预测模型，输出1-33每个数字出现的概率"""
    
    def __init__(self, input_size):
        super(ProbabilityLSTM, self).__init__()
        
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=LSTM_HIDDEN_SIZE,
            num_layers=LSTM_NUM_LAYERS,
            batch_first=True,
            dropout=DROPOUT_RATE if LSTM_NUM_LAYERS > 1 else 0
        )
        
        self.fc = nn.Linear(LSTM_HIDDEN_SIZE, NUM_NUMBERS)
        self.sigmoid = nn.Sigmoid()
    
    def forward(self, x):
        _, (h_n, _) = self.lstm(x)
        last_hidden = h_n[-1]
        return self.sigmoid(self.fc(last_hidden))


# ================= 训练模块（带时间统计和详细日志）=================
def train_model(model, train_loader, val_loader, criterion, optimizer, epochs, device):
    """训练LSTM模型（带早停机制和学习率调度）"""
    model.train()
    total_train_start = time.time()
    
    print(f"\n{'=' * 60}")
    print(f"开始训练模型 (设备: {device})")
    print(f"训练集: {len(train_loader.dataset)} 样本, 验证集: {len(val_loader.dataset)} 样本")
    print(f"批次大小: {BATCH_SIZE}, 特征维度: {train_loader.dataset[0][0].shape[-1]}")
    print(f"{'=' * 60}")
    
    best_val_loss = float('inf')
    patience_counter = 0
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, 'min', factor=LR_SCHEDULER_FACTOR, patience=LR_SCHEDULER_PATIENCE
    )
    
    for epoch in range(epochs):
        epoch_start = time.time()
        total_loss = 0.0
        batch_count = 0
        
        for inputs, targets in train_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item() * inputs.size(0)
            batch_count += 1
        
        avg_loss = total_loss / len(train_loader.dataset)
        epoch_time = time.time() - epoch_start
        
        if (epoch + 1) % VAL_FREQUENCY == 0:
            val_start = time.time()
            model.eval()
            val_loss = 0.0
            
            with torch.no_grad():
                for inputs, targets in val_loader:
                    inputs, targets = inputs.to(device), targets.to(device)
                    outputs = model(inputs)
                    val_loss += criterion(outputs, targets).item() * inputs.size(0)
            
            val_loss /= len(val_loader.dataset)
            scheduler.step(val_loss)
            val_time = time.time() - val_start
            
            current_lr = optimizer.param_groups[0]['lr']
            print(f"\nEpoch [{epoch+1}/{epochs}]")
            print(f"  训练损失: {avg_loss:.6f}, 训练时间: {epoch_time:.1f}s")
            print(f"  验证损失: {val_loss:.6f}, 验证时间: {val_time:.1f}s")
            print(f"  当前学习率: {current_lr:.6f}")
            
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0
                print(f"  ✓ 验证损失下降! 最佳损失: {best_val_loss:.6f}")
            else:
                patience_counter += 1
                print(f"  ⚠ 验证损失未下降, 耐心值: {patience_counter}/{EARLY_STOP_PATIENCE}")
                
                if patience_counter >= EARLY_STOP_PATIENCE:
                    print(f"  ✋ 早停触发, 在 Epoch {epoch+1} 停止训练")
                    break
            
            model.train()
    
    total_train_time = time.time() - total_train_start
    print(f"\n训练完成! 总训练时间: {total_train_time:.2f}秒")
    
    return model


# ================= 预测模块 =================
def predict_probabilities(model, df, feature_cols, window_size, device):
    """预测下一期每个数字出现的概率"""
    model.eval()
    
    recent_data = df[feature_cols].tail(window_size).values
    input_tensor = torch.tensor(recent_data, dtype=torch.float32).unsqueeze(0).to(device)
    
    with torch.no_grad():
        probabilities = model(input_tensor).cpu().numpy()[0]
    
    return probabilities


# ================= 结果保存模块 =================
def save_results(probabilities, output_dir="C:/Users/lw25622/ML/SSQ"):
    """将预测结果保存到以当前日期命名的文件中"""
    today = datetime.now().strftime("%Y%m%d")
    output_file = Path(output_dir) / f"predict_{today}.csv"
    
    result_df = pd.DataFrame({
        'Number': range(1, NUM_NUMBERS + 1),
        'Probability': probabilities,
        'Rank': np.argsort(probabilities)[::-1] + 1
    })
    
    result_df = result_df.sort_values('Probability', ascending=False).reset_index(drop=True)
    result_df.to_csv(output_file, index=False, encoding='utf-8-sig')
    print(f"\n预测结果已保存至: {output_file}")
    
    return str(output_file)


# ================= 评估模块 =================
def evaluate_model(model, test_loader, device):
    """评估模型在测试集上的性能"""
    start_time = time.time()
    model.eval()
    all_targets = []
    all_preds = []
    
    with torch.no_grad():
        for inputs, targets in test_loader:
            inputs = inputs.to(device)
            outputs = model(inputs)
            all_targets.extend(targets.numpy())
            all_preds.extend(outputs.cpu().numpy())
    
    all_targets = np.array(all_targets)
    all_preds = np.array(all_preds)
    
    avg_accuracy = 0.0
    for i in range(len(all_targets)):
        top6_indices = np.argsort(all_preds[i])[::-1][:6]
        target_indices = np.where(all_targets[i] == 1)[0]
        correct = len(set(top6_indices) & set(target_indices))
        avg_accuracy += correct / 6.0
    
    avg_accuracy /= len(all_targets)
    elapsed = time.time() - start_time
    
    print(f"\n{'=' * 60}")
    print("模型评估结果")
    print(f"测试集样本数: {len(all_targets)}")
    print(f"平均Top-6命中率: {avg_accuracy:.4f}")
    print(f"评估耗时: {elapsed:.2f}秒")
    print(f"{'=' * 60}")


# ================= 主程序入口 =================
def main():
    """主程序入口，执行完整的训练和预测流程"""
    total_start = time.time()
    
    print("=" * 60)
    print(f"LSTM 红球概率预测系统 - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 60)
    
    # 1. 加载数据
    t_start = time.time()
    data_path = Path("C:/Users/lw25622/ML/SSQ/1.csv")
    print(f"\n正在加载数据: {data_path}")
    
    try:
        df = pd.read_csv(data_path)
        elapsed = time.time() - t_start
        print(f"✓ 数据加载成功, 共 {len(df)} 期记录, 耗时: {elapsed:.2f}秒")
    except FileNotFoundError:
        print(f"✗ 错误: 未找到数据文件 {data_path}")
        return
    
    # 2. 计算特征
    df = calculate_all_features(df)
    
    # 3. 准备数据集
    t_start = time.time()
    feature_cols = extract_feature_columns(df)
    print(f"\n特征列数量: {len(feature_cols)}")
    
    X, y = create_sliding_windows(df, feature_cols, WINDOW_SIZE)
    
    # 4. 数据标准化
    scaler = StandardScaler()
    X = scaler.fit_transform(X.reshape(-1, X.shape[-1])).reshape(X.shape)
    elapsed = time.time() - t_start
    print(f"数据预处理完成, 耗时: {elapsed:.2f}秒")
    
    # 5. 划分训练集、验证集和测试集
    t_start = time.time()
    X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=TEST_SIZE, random_state=42, shuffle=False)
    X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=VAL_SIZE, random_state=42, shuffle=False)
    
    train_dataset = LSTMDataset(X_train, y_train)
    val_dataset = LSTMDataset(X_val, y_val)
    test_dataset = LSTMDataset(X_test, y_test)
    
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)
    
    elapsed = time.time() - t_start
    print(f"数据集划分完成: 训练集 {len(train_dataset)} | 验证集 {len(val_dataset)} | 测试集 {len(test_dataset)}, 耗时: {elapsed:.2f}秒")
    
    # 6. 构建模型
    t_start = time.time()
    input_size = len(feature_cols)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = ProbabilityLSTM(input_size=input_size).to(device)
    
    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    elapsed = time.time() - t_start
    print(f"\n模型构建完成, 总参数: {total_params}, 可训练参数: {trainable_params}, 耗时: {elapsed:.2f}秒")
    print(f"模型结构: LSTM({input_size}→{LSTM_HIDDEN_SIZE}, {LSTM_NUM_LAYERS}层) → FC({LSTM_HIDDEN_SIZE}→{NUM_NUMBERS})")
    
    # 7. 训练模型
    model = train_model(model, train_loader, val_loader, criterion, optimizer, EPOCHS, device)
    
    # 8. 评估模型
    evaluate_model(model, test_loader, device)
    
    # 9. 预测下一期概率
    t_start = time.time()
    print("\n" + "=" * 60)
    print("开始预测下一期数字出现概率...")
    
    recent_features = df[feature_cols].tail(WINDOW_SIZE).values
    recent_features = scaler.transform(recent_features.reshape(-1, input_size)).reshape(WINDOW_SIZE, input_size)
    probabilities = predict_probabilities(model, pd.DataFrame(recent_features, columns=feature_cols), 
                                          feature_cols, WINDOW_SIZE, device)
    
    elapsed = time.time() - t_start
    print(f"预测完成, 耗时: {elapsed:.2f}秒")
    
    # 10. 输出预测结果
    print("\n数字 1-33 出现概率预测结果（按概率降序）:")
    print("-" * 60)
    print(f"{'Type':<6} {'数字':<6} {'概率':<10}")
    print("-" * 60)
    
    sorted_indices = np.argsort(probabilities)[::-1]
    for rank, idx in enumerate(sorted_indices, 1):
        number = idx + 1
        prob = probabilities[idx]
        print(f"{'LSTM':<6} {number:<6} {prob:<10.6f}")
    
    # 11. 保存结果
    save_results(probabilities)
    
    # 12. 总耗时统计
    total_time = time.time() - total_start
    print(f"\n{'=' * 60}")
    print(f"处理完成!")
    print(f"总耗时: {total_time:.2f}秒")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    main()