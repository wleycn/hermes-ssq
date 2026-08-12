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

'''
长短期记忆网络LSTM通过引入门控机制 遗忘门、输入门、输出门 有效解决了梯度消失问题 能够学习长期依赖
'''

# ================= 全局参数配置 =================
# ========== 数据相关参数 ==========
WINDOW_SIZE = 330         # 滑动窗口大小，使用过去多少期数据预测下一期
RED_NUMBERS = 33          # 红球数字范围（1-33）
BLUE_NUMBERS = 16         # 蓝球数字范围（1-16）
RECENT_WINDOW = 330      # 计算近期频率特征的窗口大小

TARGET_COLS = ['Red1', 'Red2', 'Red3', 'Red4', 'Red5', 'Red6', 'Blue1']  # 目标列名（红球6个位置+蓝球1个位置）
RED_COLS = ['Red1', 'Red2', 'Red3', 'Red4', 'Red5', 'Red6']             # 红球列名
BLUE_COLS = ['Blue1']                                                   # 蓝球列名

# ========== 训练相关参数 ==========
BATCH_SIZE = 240           # 批次大小，每批处理的样本数，增大可提高GPU/CPU利用率
EPOCHS = 256              # 最大训练轮数，实际会被早停截断
LEARNING_RATE = 0.0003    # 初始学习率，控制参数更新步长

# ========== 模型结构参数 ==========
LSTM_HIDDEN_SIZE = 32     # LSTM隐藏层神经元数量，越大模型容量越大但训练越慢
LSTM_NUM_LAYERS = 1       # LSTM层数，多层可捕捉更复杂模式但训练更慢
DROPOUT_RATE = 0.1        # Dropout率，防止过拟合，0表示不使用dropout
FC_HIDDEN_SIZE = 32       # 全连接层神经元数量

# ========== 正则化参数 ==========
L2_REG = 1e-4             # L2正则化系数（weight_decay），用于防止过拟合

# ========== 早停与学习率调度参数 ==========
EARLY_STOP_PATIENCE = 3   # 早停耐心值，验证集损失连续多少轮不下降则停止训练
LR_SCHEDULER_FACTOR = 0.5 # 学习率衰减因子，验证集损失不下降时乘以该值
LR_SCHEDULER_PATIENCE=10  # 学习率调度耐心值，连续多少轮不下降则衰减学习率
VAL_FREQUENCY = 10        # 验证频率，每多少个epoch进行一次验证

# ========== 数据划分参数 ==========
TEST_SIZE = 0.2           # 测试集比例，从总数据中划分多少作为测试集
VAL_SIZE = 0.5            # 验证集比例，从测试集中划分多少作为验证集（剩余为测试集）


# ================= 特征工程模块（完整向量化版）=================
def calc_odd_even_features(df):
    for i, col in enumerate(TARGET_COLS):
        df[f'Odd_Even_{i+1}'] = (df[col] % 2).astype(np.float32)
    return df


def calc_recent_frequency(df):
    """计算最近RECENT_WINDOW期内每个数字的出现频率（完全向量化）"""
    red_data = df[RED_COLS].values
    
    for num in range(1, RED_NUMBERS + 1):
        mask = (red_data == num).astype(np.float32).sum(axis=1)
        padded = np.zeros(len(df) + RECENT_WINDOW - 1, dtype=np.float32)
        padded[RECENT_WINDOW - 1:] = mask
        df[f'Recent_Freq_Red_{num}'] = np.convolve(padded, np.ones(RECENT_WINDOW, dtype=np.float32), mode='valid')
    
    blue_data = df['Blue1'].values
    for num in range(1, BLUE_NUMBERS + 1):
        mask = (blue_data == num).astype(np.float32)
        padded = np.zeros(len(df) + RECENT_WINDOW - 1, dtype=np.float32)
        padded[RECENT_WINDOW - 1:] = mask
        df[f'Recent_Freq_Blue_{num}'] = np.convolve(padded, np.ones(RECENT_WINDOW, dtype=np.float32), mode='valid')
    
    return df


def calc_last_appearance(df):
    """计算每个数字距离上次出现的期数（完全向量化）"""
    red_data = df[RED_COLS].values
    
    for num in range(1, RED_NUMBERS + 1):
        mask = (red_data == num).any(axis=1).astype(np.float32)
        if mask.sum() == 0:
            df[f'Last_Appear_Red_{num}'] = np.arange(1, len(df) + 1, dtype=np.float32)
        else:
            positions = np.where(mask == 1)[0]
            expanded = np.zeros(len(df), dtype=np.float32)
            expanded[positions] = positions.astype(np.float32) + 1
            last_pos = np.maximum.accumulate(expanded)
            last_pos[last_pos == 0] = np.nan
            result = np.arange(1, len(df) + 1, dtype=np.float32) - last_pos
            df[f'Last_Appear_Red_{num}'] = np.nan_to_num(result, nan=np.arange(1, len(df) + 1, dtype=np.float32))
    
    blue_data = df['Blue1'].values
    for num in range(1, BLUE_NUMBERS + 1):
        mask = (blue_data == num).astype(np.float32)
        if mask.sum() == 0:
            df[f'Last_Appear_Blue_{num}'] = np.arange(1, len(df) + 1, dtype=np.float32)
        else:
            positions = np.where(mask == 1)[0]
            expanded = np.zeros(len(df), dtype=np.float32)
            expanded[positions] = positions.astype(np.float32) + 1
            last_pos = np.maximum.accumulate(expanded)
            last_pos[last_pos == 0] = np.nan
            result = np.arange(1, len(df) + 1, dtype=np.float32) - last_pos
            df[f'Last_Appear_Blue_{num}'] = np.nan_to_num(result, nan=np.arange(1, len(df) + 1, dtype=np.float32))
    
    return df


def calc_poisson_features(df, window_size=100):
    """计算每个号码在滑动窗口内的泊松分布概率（完全向量化）"""
    red_data = df[RED_COLS].values
    
    for num in range(1, RED_NUMBERS + 1):
        mask = (red_data == num).astype(np.float32).sum(axis=1)
        padded = np.zeros(len(df) + window_size - 1, dtype=np.float32)
        padded[window_size - 1:] = mask
        lam = np.convolve(padded, np.ones(window_size, dtype=np.float32), mode='valid') / window_size
        df[f'Poisson_Red_{num}'] = poisson.pmf(1, lam)
    
    blue_data = df['Blue1'].values
    for num in range(1, BLUE_NUMBERS + 1):
        mask = (blue_data == num).astype(np.float32)
        padded = np.zeros(len(df) + window_size - 1, dtype=np.float32)
        padded[window_size - 1:] = mask
        lam = np.convolve(padded, np.ones(window_size, dtype=np.float32), mode='valid') / window_size
        df[f'Poisson_Blue_{num}'] = poisson.pmf(1, lam)
    
    return df


def calc_normal_features(df):
    """计算红球和值及蓝球值的正态分布参数特征"""
    df['Red_Sum'] = df[RED_COLS].sum(axis=1)
    
    rolling = df['Red_Sum'].rolling(window=50, min_periods=10)
    df['Red_Sum_Mean'] = rolling.mean().fillna(df['Red_Sum'].mean())
    df['Red_Sum_Std'] = rolling.std().fillna(df['Red_Sum'].std())
    df['Red_Sum_Skew'] = rolling.skew().fillna(0)
    df['Red_Sum_Kurt'] = rolling.kurt().fillna(0)
    
    rolling = df['Blue1'].rolling(window=50, min_periods=10)
    df['Blue_Mean'] = rolling.mean().fillna(df['Blue1'].mean())
    df['Blue_Std'] = rolling.std().fillna(df['Blue1'].std())
    df['Blue_Skew'] = rolling.skew().fillna(0)
    df['Blue_Kurt'] = rolling.kurt().fillna(0)
    
    return df


def calc_entropy_features(df, window_size=50):
    """计算红球和蓝球分布的信息熵特征（向量化优化）"""
    red_data = df[RED_COLS].values
    blue_data = df['Blue1'].values
    
    def compute_entropy(data, window_size):
        n = len(data)
        entropy = np.zeros(n, dtype=np.float32)
        
        if data.ndim > 1:
            data_flat = data.flatten()
            stride = data.shape[1]
        else:
            data_flat = data
            stride = 1
        
        for i in range(window_size, n):
            start = i * stride - window_size * stride
            end = i * stride
            window = data_flat[start:end]
            _, counts = np.unique(window, return_counts=True)
            probs = counts / len(window)
            entropy[i] = -np.sum(probs * np.log2(probs + 1e-10))
        
        return entropy
    
    df['Red_Entropy'] = compute_entropy(red_data, window_size)
    df['Blue_Entropy'] = compute_entropy(blue_data, window_size)
    
    return df


def calc_markov_features(df):
    """计算红球奇偶比和蓝球奇偶状态的马尔可夫链转移概率特征"""
    df['Red_Odd_Count'] = (df[RED_COLS] % 2 == 1).sum(axis=1).astype(np.int32)
    df['Blue_Odd'] = (df['Blue1'] % 2 == 1).astype(np.int32)
    
    red_states = 7
    red_counts = np.zeros((red_states, red_states), dtype=np.float32)
    red_odd_vals = df['Red_Odd_Count'].values
    
    for i in range(1, len(df)):
        red_counts[red_odd_vals[i - 1], red_odd_vals[i]] += 1
    
    with np.errstate(divide='ignore', invalid='ignore'):
        red_prob = red_counts / red_counts.sum(axis=1, keepdims=True)
        red_prob = np.nan_to_num(red_prob)
    
    for state in range(red_states):
        df[f'Markov_Prob_{state}'] = red_prob[red_odd_vals, state]
    
    blue_states = 2
    blue_counts = np.zeros((blue_states, blue_states), dtype=np.float32)
    blue_odd_vals = df['Blue_Odd'].values
    
    for i in range(1, len(df)):
        blue_counts[blue_odd_vals[i - 1], blue_odd_vals[i]] += 1
    
    with np.errstate(divide='ignore', invalid='ignore'):
        blue_prob = blue_counts / blue_counts.sum(axis=1, keepdims=True)
        blue_prob = np.nan_to_num(blue_prob)
    
    for state in range(blue_states):
        df[f'Blue_Markov_Prob_{state}'] = blue_prob[blue_odd_vals, state]
    
    return df


def calc_position_features(df):
    """计算每个位置号码的统计特征"""
    for i, col in enumerate(TARGET_COLS):
        rolling = df[col].rolling(window=30, min_periods=5)
        df[f'Pos_{i+1}_Mean'] = rolling.mean().fillna(df[col].mean())
        df[f'Pos_{i+1}_Std'] = rolling.std().fillna(df[col].std())
        df[f'Pos_{i+1}_Recent'] = df[col].rolling(window=3, min_periods=1).mean().fillna(df[col].mean())
    
    return df


def calculate_all_features(df):
    """计算所有特征的主入口函数（带时间统计）"""
    start_time = time.time()
    
    # 奇偶标志特征
    t_start = time.time()
    df = calc_odd_even_features(df)
    elapsed = time.time() - t_start
    print(f"  [耗时 {elapsed:.2f}s] 奇偶标志特征")
    
    # 近期频率特征
    t_start = time.time()
    df = calc_recent_frequency(df)
    elapsed = time.time() - t_start
    print(f"  [耗时 {elapsed:.2f}s] 近期频率特征")
    
    # 上次出现位置特征
    t_start = time.time()
    df = calc_last_appearance(df)
    elapsed = time.time() - t_start
    print(f"  [耗时 {elapsed:.2f}s] 上次出现位置特征")
    
    # 泊松分布特征
    t_start = time.time()
    df = calc_poisson_features(df)
    elapsed = time.time() - t_start
    print(f"  [耗时 {elapsed:.2f}s] 泊松分布特征")
    
    # 正态分布特征
    t_start = time.time()
    df = calc_normal_features(df)
    elapsed = time.time() - t_start
    print(f"  [耗时 {elapsed:.2f}s] 正态分布特征")
    
    # 信息熵特征
    t_start = time.time()
    df = calc_entropy_features(df)
    elapsed = time.time() - t_start
    print(f"  [耗时 {elapsed:.2f}s] 信息熵特征")
    
    # 马尔可夫链特征
    t_start = time.time()
    df = calc_markov_features(df)
    elapsed = time.time() - t_start
    print(f"  [耗时 {elapsed:.2f}s] 马尔可夫链特征") 
    
    # 位置统计特征
    t_start = time.time()
    df = calc_position_features(df)
    elapsed = time.time() - t_start
    print(f"  [耗时 {elapsed:.2f}s] 位置统计特征")
    
    total_time = time.time() - start_time
    print(f"\n特征工程总耗时: {total_time:.2f}秒")
    print(f"有效记录数: {len(df)}")
    
    return df.dropna().reset_index(drop=True)


# ================= 数据准备模块 =================
def extract_feature_columns(df):
    exclude_cols = ['dDate', 'dNum', 'yNum', 'mNum', 'Red_Sum', 'Red_Odd_Count', 'Blue_Odd'] + TARGET_COLS
    feature_cols = [col for col in df.columns if col not in exclude_cols]
    return feature_cols


def create_sliding_windows(df, feature_cols, window_size):
    """创建滑动窗口数据集（带时间统计）"""
    start_time = time.time()
    
    n = len(df)
    X_data = df[feature_cols].values
    
    X = np.zeros((n - window_size, window_size, len(feature_cols)), dtype=np.float32)
    y = np.zeros((n - window_size, RED_NUMBERS + BLUE_NUMBERS), dtype=np.float32)
    
    for i in range(n - window_size):
        X[i] = X_data[i:i + window_size]
        
        red_nums = df[RED_COLS].iloc[i + window_size].values.astype(np.int32)
        blue_num = df[BLUE_COLS].iloc[i + window_size].values[0].astype(np.int32)
        
        y[i, red_nums - 1] = 1.0
        y[i, RED_NUMBERS + blue_num - 1] = 1.0
    
    elapsed = time.time() - start_time
    print(f"滑动窗口生成完成 - X: {X.shape}, y: {y.shape}, 耗时: {elapsed:.2f}秒")
    
    return X, y


# ================= 数据集与模型模块 =================
class LSTMDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.float32)
    
    def __len__(self):
        return len(self.X)
    
    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


class HybridLSTM(nn.Module):
    """混合LSTM概率预测模型（完整结构）"""
    
    def __init__(self, input_size):
        super(HybridLSTM, self).__init__()
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=LSTM_HIDDEN_SIZE,
            num_layers=LSTM_NUM_LAYERS,
            batch_first=True,
            dropout=DROPOUT_RATE
        )
        
        self.fc1 = nn.Linear(LSTM_HIDDEN_SIZE, FC_HIDDEN_SIZE)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(DROPOUT_RATE)
        
        self.red_fc = nn.Linear(FC_HIDDEN_SIZE, RED_NUMBERS)
        self.blue_fc = nn.Linear(FC_HIDDEN_SIZE, BLUE_NUMBERS)
        self.sigmoid = nn.Sigmoid()
    
    def forward(self, x):
        _, (h_n, _) = self.lstm(x)
        last_hidden = h_n[-1]
        
        out = self.fc1(last_hidden)
        out = self.relu(out)
        out = self.dropout(out)
        
        red_out = self.sigmoid(self.red_fc(out))
        blue_out = self.sigmoid(self.blue_fc(out))
        
        return torch.cat([red_out, blue_out], dim=1)


# ================= 训练模块（带早停机制）=================
def train_model(model, train_loader, val_loader, criterion, optimizer, epochs, device, patience=EARLY_STOP_PATIENCE):
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
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', factor=LR_SCHEDULER_FACTOR, patience=LR_SCHEDULER_PATIENCE)
    
    for epoch in range(epochs):
        epoch_start = time.time()
        total_loss = 0.0
        
        for inputs, targets in train_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item() * inputs.size(0)
        
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
                print(f"  ⚠ 验证损失未下降, 耐心值: {patience_counter}/{patience}")
                
                if patience_counter >= patience:
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
        probs = model(input_tensor).cpu().numpy()[0]
    
    return probs[:RED_NUMBERS], probs[RED_NUMBERS:]


# ================= 结果保存模块 =================
def save_results(red_probs, blue_probs, output_dir="C:/Users/lw25622/ML/SSQ"):
    """将预测结果保存到以当前日期命名的文件中"""
    today = datetime.now().strftime("%Y%m%d")
    output_file = Path(output_dir) / f"predict_all_{today}.csv"
    
    red_df = pd.DataFrame({
        'Type': 'Red',
        'Number': range(1, RED_NUMBERS + 1),
        'Probability': red_probs,
        'Rank': np.argsort(red_probs)[::-1] + 1
    })
    
    blue_df = pd.DataFrame({
        'Type': 'Blue',
        'Number': range(1, BLUE_NUMBERS + 1),
        'Probability': blue_probs,
        'Rank': np.argsort(blue_probs)[::-1] + 1
    })
    
    result_df = pd.concat([red_df, blue_df], ignore_index=True)
    result_df = result_df.sort_values(['Type', 'Probability'], ascending=[True, False]).reset_index(drop=True)
    result_df.to_csv(output_file, index=False, encoding='utf-8-sig')
    
    print(f"\n预测结果已保存至: {output_file}")
    return str(output_file)


# ================= 评估模块 =================
def evaluate_model(model, test_loader, device):
    """评估模型在测试集上的性能"""
    start_time = time.time()
    model.eval()
    red_acc = 0.0
    blue_acc = 0.0
    count = 0
    
    with torch.no_grad():
        for inputs, targets in test_loader:
            inputs = inputs.to(device)
            outputs = model(inputs).cpu().numpy()
            targets = targets.numpy()
            
            for i in range(len(outputs)):
                top6 = np.argsort(outputs[i, :RED_NUMBERS])[::-1][:6]
                target_red = np.where(targets[i, :RED_NUMBERS] == 1)[0]
                red_acc += len(set(top6) & set(target_red)) / 6.0
                
                top1 = np.argsort(outputs[i, RED_NUMBERS:])[::-1][:1]
                target_blue = np.where(targets[i, RED_NUMBERS:] == 1)[0]
                blue_acc += len(set(top1) & set(target_blue)) / 1.0
                count += 1
    
    elapsed = time.time() - start_time
    
    print(f"\n{'=' * 60}")
    print("模型评估结果")
    print(f"测试集样本数: {count}")
    print(f"红球平均Top-6命中率: {red_acc/count:.4f}")
    print(f"蓝球平均Top-1命中率: {blue_acc/count:.4f}")
    print(f"评估耗时: {elapsed:.2f}秒")
    print(f"{'=' * 60}")


# ================= 主程序入口 =================
def main():
    """主程序入口，执行完整的训练和预测流程"""
    total_start = time.time()
    
    print("=" * 60)
    print(f"LSTM 双色球(红球+蓝球)概率预测系统 - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
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
    print("\n" + "=" * 60)
    print("开始特征工程...")
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
    model = HybridLSTM(input_size=input_size).to(device)
    
    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=L2_REG)
    
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    elapsed = time.time() - t_start
    print(f"\n模型构建完成, 总参数: {total_params}, 可训练参数: {trainable_params}, 耗时: {elapsed:.2f}秒")
    print(f"模型结构: HybridLSTM({input_size}→{LSTM_HIDDEN_SIZE}, {LSTM_NUM_LAYERS}层) → FC({FC_HIDDEN_SIZE}→{RED_NUMBERS+BLUE_NUMBERS})")
    
    # 7. 训练模型
    model = train_model(model, train_loader, val_loader, criterion, optimizer, EPOCHS, device)
    
    # 8. 评估模型
    evaluate_model(model, test_loader, device)
    
    # 9. 预测下一期概率
    t_start = time.time()
    print("\n" + "=" * 60)
    print("开始预测下一期开奖结果...")
    
    recent_features = df[feature_cols].tail(WINDOW_SIZE).values
    recent_features = scaler.transform(recent_features.reshape(-1, input_size)).reshape(WINDOW_SIZE, input_size)
    red_probs, blue_probs = predict_probabilities(model, pd.DataFrame(recent_features, columns=feature_cols), 
                                                  feature_cols, WINDOW_SIZE, device)
    
    elapsed = time.time() - t_start
    print(f"预测完成, 耗时: {elapsed:.2f}秒")
    
    recent_entropy = df['Red_Entropy'].iloc[-1]
    entropy_threshold = 4.0
    if recent_entropy > entropy_threshold:
        print(f"\n裁判端警告：信息熵 {recent_entropy:.4f} > {entropy_threshold}，系统高混乱状态！")
    
    red_top6 = sorted(np.argsort(red_probs)[::-1][:6] + 1)
    blue_top1 = int(np.argsort(blue_probs)[::-1][:1][0] + 1)
    
    print(f"\n预测结果 (基于过去{WINDOW_SIZE}期):")
    print("-" * 70)
    print(f"红球预测: {np.array(red_top6)}")
    print(f"蓝球预测: {blue_top1}")
    
    # 10. 保存结果
    save_results(red_probs, blue_probs)
    
    # 11. 总耗时统计
    total_time = time.time() - total_start
    print(f"\n{'=' * 60}")
    print(f"处理完成!")
    print(f"总耗时: {total_time:.2f}秒")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    main()
