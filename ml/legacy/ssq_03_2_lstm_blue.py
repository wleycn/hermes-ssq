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
NUM_NUMBERS = 16           # 蓝球数字范围（1-16）
RECENT_WINDOW = 512        # 计算近期频率特征的窗口大小
TARGET_COLS = ['Blue1']    # 目标列名，即蓝球列

# ========== 训练相关参数 ==========
BATCH_SIZE = 128           # 批次大小，每批处理的样本数，增大可提高GPU/CPU利用率
EPOCHS = 256              # 最大训练轮数，实际会被早停截断
LEARNING_RATE = 0.001     # 初始学习率，控制参数更新步长

# ========== 模型结构参数 ==========
LSTM_HIDDEN_SIZE = 32      # LSTM隐藏层神经元数量，越大模型容量越大但训练越慢
LSTM_NUM_LAYERS = 2        # LSTM层数，多层可捕捉更复杂模式但训练更慢
DROPOUT_RATE = 0.05        # Dropout率，防止过拟合，0表示不使用dropout

# ========== 早停与学习率调度参数 ==========
EARLY_STOP_PATIENCE = 7   # 早停耐心值，验证集损失连续多少轮不下降则停止训练
LR_SCHEDULER_FACTOR = 0.5  # 学习率衰减因子，验证集损失不下降时乘以该值
LR_SCHEDULER_PATIENCE = 3  # 学习率调度耐心值，连续多少轮不下降则衰减学习率
VAL_FREQUENCY = 10         # 验证频率，每多少个epoch进行一次验证

# ========== 数据划分参数 ==========
TEST_SIZE = 0.2            # 测试集比例，从总数据中划分多少作为测试集
VAL_SIZE = 0.3            # 验证集比例，从测试集中划分多少作为验证集（剩余为测试集）


# ================= 特征工程模块（向量化优化）=================
def calc_odd_even_features(df):
    df['Blue_Odd_Even'] = (df['Blue1'] % 2).astype(np.float32)
    return df


def calc_recent_frequency(df):
    """计算最近RECENT_WINDOW期内每个蓝球数字的出现频率（使用rolling优化）"""
    for num in range(1, NUM_NUMBERS + 1):
        df[f'Recent_Freq_{num}'] = (df['Blue1'] == num).rolling(window=RECENT_WINDOW, min_periods=1).sum()
    
    return df


def calc_last_appearance(df):
    """计算每个蓝球数字距离上次出现的期数（向量化）"""
    blue_data = df['Blue1'].values
    
    for num in range(1, NUM_NUMBERS + 1):
        mask = (blue_data == num).astype(np.float32)
        if mask.sum() == 0:
            df[f'Last_Appear_{num}'] = np.arange(1, len(df) + 1, dtype=np.float32)
        else:
            positions = np.where(mask == 1)[0]
            expanded = np.zeros(len(df), dtype=np.float32)
            expanded[positions] = positions.astype(np.float32) + 1
            last_pos = np.maximum.accumulate(expanded)
            last_pos[last_pos == 0] = np.nan
            result = np.arange(1, len(df) + 1, dtype=np.float32) - last_pos
            df[f'Last_Appear_{num}'] = np.nan_to_num(result, nan=np.arange(1, len(df) + 1, dtype=np.float32))
    
    return df


def calc_poisson_features(df, window_size=100):
    """计算每个蓝球数字在滑动窗口内的泊松分布概率（使用rolling优化）"""
    for num in range(1, NUM_NUMBERS + 1):
        freq_col = (df['Blue1'] == num).rolling(window=window_size, min_periods=1).sum()
        lam = freq_col / window_size
        df[f'Poisson_{num}'] = poisson.pmf(1, lam)
    
    return df


def calc_normal_features(df):
    df['Blue_Mean'] = df['Blue1'].rolling(window=50, min_periods=10).mean().fillna(df['Blue1'].mean())
    df['Blue_Std'] = df['Blue1'].rolling(window=50, min_periods=10).std().fillna(df['Blue1'].std())
    df['Blue_Skew'] = df['Blue1'].rolling(window=50, min_periods=10).skew().fillna(0)
    df['Blue_Kurt'] = df['Blue1'].rolling(window=50, min_periods=10).kurt().fillna(0)
    
    return df


def calc_entropy_features(df, window_size=50):
    """计算蓝球分布的信息熵特征（优化版）"""
    blue_data = df['Blue1'].values
    n = len(df)
    entropy = np.zeros(n, dtype=np.float32)
    
    for i in range(window_size, n):
        window = blue_data[i - window_size:i]
        _, counts = np.unique(window, return_counts=True)
        probs = counts / len(window)
        entropy[i] = -np.sum(probs * np.log2(probs + 1e-10))
    
    df['Entropy'] = entropy
    return df


def calc_markov_features(df):
    """计算蓝球奇偶状态的马尔可夫链转移概率特征（优化版）"""
    df['Blue_Odd'] = (df['Blue1'] % 2 == 1).astype(np.int32)
    
    states = 2
    blue_odd_vals = df['Blue_Odd'].values
    
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
    df['Blue_Pos_Mean'] = df['Blue1'].rolling(window=30, min_periods=5).mean().fillna(df['Blue1'].mean())
    df['Blue_Pos_Std'] = df['Blue1'].rolling(window=30, min_periods=5).std().fillna(df['Blue1'].std())
    df['Blue_Pos_Recent'] = df['Blue1'].rolling(window=3, min_periods=1).mean().fillna(df['Blue1'].mean())
    
    return df


def calculate_all_features(df):
    """计算所有特征的主入口函数（带时间统计）"""
    start_time = time.time()
    feature_times = {}
    
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
    
    total_time = time.time() - start_time
    print(f"\n特征工程总耗时: {total_time:.2f}秒")
    print(f"有效记录数: {len(df)}")
    
    return df.dropna().reset_index(drop=True)


# ================= 数据准备模块（优化版）=================
def extract_feature_columns(df):
    exclude_cols = ['dDate', 'dNum', 'yNum', 'mNum', 'Blue1', 'Blue_Odd'] + TARGET_COLS
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
        blue_num = df[TARGET_COLS].iloc[i + window_size].values[0].astype(np.int32)
        y[i, blue_num - 1] = 1.0
    
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


class ProbabilityLSTM(nn.Module):
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


# ================= 训练模块（带早停机制）=================
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
    model.eval()
    
    recent_data = df[feature_cols].tail(window_size).values
    input_tensor = torch.tensor(recent_data, dtype=torch.float32).unsqueeze(0).to(device)
    
    with torch.no_grad():
        probabilities = model(input_tensor).cpu().numpy()[0]
    
    return probabilities


# ================= 结果保存模块 =================
def save_results(probabilities, output_dir="C:/Users/lw25622/ML/SSQ"):
    today = datetime.now().strftime("%Y%m%d")
    output_file = Path(output_dir) / f"blue_predict_{today}.csv"
    
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
        top1_indices = np.argsort(all_preds[i])[::-1][:1]
        target_indices = np.where(all_targets[i] == 1)[0]
        avg_accuracy += len(set(top1_indices) & set(target_indices)) / 1.0
    
    avg_accuracy /= len(all_targets)
    elapsed = time.time() - start_time
    
    print(f"\n{'=' * 60}")
    print("模型评估结果")
    print(f"测试集样本数: {len(all_targets)}")
    print(f"平均Top-1命中率: {avg_accuracy:.4f}")
    print(f"评估耗时: {elapsed:.2f}秒")
    print(f"{'=' * 60}")


# ================= 主程序入口 =================
def main():
    """主程序入口，执行完整的训练和预测流程"""
    total_start = time.time()
    
    print("=" * 60)
    print(f"LSTM 蓝球概率预测系统 - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
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
    print("开始预测下一期蓝球数字出现概率...")
    
    recent_features = df[feature_cols].tail(WINDOW_SIZE).values
    recent_features = scaler.transform(recent_features.reshape(-1, input_size)).reshape(WINDOW_SIZE, input_size)
    probabilities = predict_probabilities(model, pd.DataFrame(recent_features, columns=feature_cols), 
                                          feature_cols, WINDOW_SIZE, device)
    
    elapsed = time.time() - t_start
    print(f"预测完成, 耗时: {elapsed:.2f}秒")
    
    # 10. 输出预测结果
    print("\n蓝球数字 1-16 出现概率预测结果（按概率降序）:")
    print("-" * 60)
    print(f"{'排名':<6} {'数字':<6} {'概率':<10}")
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