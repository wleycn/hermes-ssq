import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from pathlib import Path
import warnings
from datetime import datetime
from scipy.stats import poisson, norm
import time

warnings.filterwarnings('ignore')

# ================= 全局参数配置 =================
# ========== 数据相关参数 ==========
RED_STATES = 28                                                           # 每个红球位置的有效状态数（位置i的红球范围：i~33-i，共28个有效值）
BLUE_STATES = 16                                                          # 蓝球的有效状态数（1~16）
RED_NUMBERS = 33                                                          # 红球数字范围（1-33）
BLUE_NUMBERS = 16                                                         # 蓝球数字范围（1-16）
TARGET_COLS = ['Red1', 'Red2', 'Red3', 'Red4', 'Red5', 'Red6', 'Blue1']  # 目标列名（红球6个位置+蓝球1个位置）
REGRESSION_TARGETS = [                                                    # 回归任务目标列
    'Next_Sum', 'Next_OddRatio', 'Next_BigRatio', 
    'Next_Hot', 'Next_Cold', 'Next_Max_Omission', 'Next_Avg_Omission'
]

# ========== 训练相关参数 ==========
RETRAIN_MODEL = 'Y'       # 是否重新训练模型 ('Y'=是, 'N'=否，加载已有模型)
EPOCHS = 330              # 最大训练轮数
BATCH_SIZE = 240          # 批次大小，每批处理的样本数
LEARNING_RATE = 0.001    # 初始学习率，控制参数更新步长

# ========== 滑动窗口参数 ========== 
WINDOW_SIZE = 33          # 滑动窗口大小，使用过去多少期数据预测下一期
WINDOW_STEP = 1           # 滑动步长，每次移动多少期数据

# ========== 模型结构参数 ==========
CNN_OUT_CHANNELS = 5     # 卷积层输出通道数（卷积核数量）
CNN_KERNEL_SIZE = 9      # 卷积核大小（时间维度上的窗口大小）
CNN_POOL_SIZE = 1         # 池化层大小
FC_HIDDEN_SIZE = 5       # 全连接层隐藏层神经元数量

# ========== 损失函数权重参数 ==========
REGRESSION_LOSS_WEIGHT = 0.01  # 回归损失在总损失中的权重（总损失 = 分类损失 + 权重 * 回归损失）

# ========== 早停与学习率调度参数 ==========
EARLY_STOP_PATIENCE = 5     # 早停耐心值，验证集损失连续多少轮不下降则停止训练
LR_SCHEDULER_FACTOR = 0.9    # 学习率衰减因子，验证集损失不下降时乘以该值
LR_SCHEDULER_PATIENCE = 2    # 学习率调度耐心值，连续多少轮不下降则衰减学习率
TRAIN_LOG_INTERVAL = 20      # 训练日志打印间隔，每多少轮打印一次训练/验证信息

# ========== 正则化参数（防止过拟合） ==========
DROPOUT_RATE = 0.3           # Dropout比例，随机丢弃神经元的概率（0-1之间）
                             # - 值越大，正则化越强，模型越不容易过拟合
                             # - 值过小（<0.2）正则化效果弱，值过大（>0.5）可能欠拟合
                             # - 推荐范围：0.2-0.5，根据模型复杂度调整
WEIGHT_DECAY = 1e-4          # L2正则化系数（权重衰减），对模型权重的惩罚项
                             # - 值越大，权重越趋向于0，模型越简单
                             # - 值过小（<1e-5）正则化效果弱，值过大（>1e-3）可能欠拟合
                             # - 推荐范围：1e-5-1e-3，与Dropout配合使用

# ========== 特征工程参数 ==========
ENTROPY_WINDOW = 60       # 计算信息熵的滑动窗口大小
NORMAL_SIGMA = 2.58       # 正态分布过滤阈值（对应99%置信区间，用于防守端异常检测）
MARKOV_STATES = 7         # 马尔可夫链状态数（奇偶/大小比的状态：0/6, 1/5, ..., 6/0，共7种）
POISSON_WINDOW = 330      # 计算泊松分布特征的滑动窗口大小

# ========== 数据划分参数 ==========
TEST_SIZE = 0.2           # 测试集比例，从总数据中划分多少作为测试集

# ========== 预测后处理参数 ==========
ENTROPY_CHAOS_THRESHOLD = 4.0  # 信息熵混乱阈值，超过此值判定为高混乱状态
SUM_CONSTRAINT_THRESHOLD = 10   # 和值修正阈值，当预测和值与当前和值的差值超过此值时进行修正
CHAOS_DAMPING_FACTOR = 0.5      # 混乱状态下的预测概率衰减因子，降低模型输出的置信度
                                 # - 值越小，衰减越明显，预测结果越趋向于平均
                                 # - 值越大，衰减越弱，保留更多模型预测信息


# ================= 特征提取工具函数 =================
def extract_feature_matrix(df):
    """提取6类特征并拼接为完整特征矩阵（训练和预测共享）

    特征构成：
    - 基础数据：7列（Red1~Red6 + Blue1）
    - One-Hot编码：33列（Last_Draw_1~33）
    - 泊松特征：49列（Poisson_R1~33 + Poisson_B1~16）
    - 正态分布参数：2列（Norm_Mean, Norm_Std）
    - 信息熵：1列（Entropy）
    - 马尔可夫链：14列（Markov_Odd_Prob_0~6 + Markov_Big_Prob_0~6）
    总计：106列

    Args:
        df: 包含所有特征的 DataFrame

    Returns:
        np.ndarray: 特征矩阵，形状 (n_rows, 106)
    """
    base_data = df[TARGET_COLS].values.astype(np.float32)
    
    last_draw_cols = [f'Last_Draw_{i}' for i in range(1, 34)]
    last_draw_data = df[last_draw_cols].values.astype(np.float32)
    
    poisson_cols = [c for c in df.columns if 'Poisson' in c]
    poisson_data = df[poisson_cols].values.astype(np.float32)
    
    norm_cols = ['Norm_Mean', 'Norm_Std']
    norm_data = df[norm_cols].values.astype(np.float32)
    
    entropy_cols = ['Entropy']
    entropy_data = df[entropy_cols].values.astype(np.float32)
    
    markov_cols = [c for c in df.columns if 'Markov' in c]
    markov_data = df[markov_cols].values.astype(np.float32)
    
    return np.hstack((base_data, last_draw_data, poisson_data, norm_data, entropy_data, markov_data))


# ================= 衍生特征计算模块 =================
def calc_base_features(df):
    """计算基础统计特征（和值、奇偶比、大小比）

    Args:
        df: 原始数据 DataFrame

    Returns:
        pd.DataFrame: 包含 Sum, OddRatio, BigRatio 列的 DataFrame
    """
    df = df.copy()
    all_reds = df[TARGET_COLS[:6]]
    df['Sum'] = all_reds.sum(axis=1)
    df['OddRatio'] = (all_reds % 2 == 1).sum(axis=1) / 6.0
    df['BigRatio'] = (all_reds >= 17).sum(axis=1) / 6.0
    return df


def calc_onehot_features(df):
    """生成上期红球号码的 One-Hot 编码特征（向量化版）

    Args:
        df: 包含红球数据的 DataFrame

    Returns:
        pd.DataFrame: 添加了 Last_Draw_1~33 列的 DataFrame
    """
    df = df.copy()
    all_reds = df[TARGET_COLS[:6]].shift(1).values
    
    for i in range(1, 34):
        df[f'Last_Draw_{i}'] = (all_reds == i).any(axis=1).astype(np.float32)
    
    return df


def calc_statistical_features(df, window_size):
    """计算 Hot/Cold/Omission 统计特征（向量化版）

    Args:
        df: 原始数据 DataFrame
        window_size: 滑动窗口大小

    Returns:
        pd.DataFrame: 添加了 Hot_Count, Cold_Count, Max_Omission, Avg_Omission 列的 DataFrame
    """
    df = df.copy()
    red_data = df[TARGET_COLS[:6]].values
    
    hot_counts = np.zeros(len(df), dtype=np.float32)
    cold_counts = np.zeros(len(df), dtype=np.float32)
    max_omissions = np.zeros(len(df), dtype=np.float32)
    avg_omissions = np.zeros(len(df), dtype=np.float32)
    
    for num in range(1, 34):
        positions = np.where(red_data == num)[0]
        last_seen = np.zeros(len(df), dtype=np.int32)
        
        ptr = 0
        for i in range(len(df)):
            if ptr < len(positions) and positions[ptr] == i:
                last_seen[i] = i
                ptr += 1
            elif ptr > 0:
                last_seen[i] = last_seen[i-1]
            else:
                last_seen[i] = -1
        
        omissions = np.arange(len(df)) - last_seen
        mask = last_seen == -1
        omissions[mask] = np.where(mask)[0] + 1
        
        max_omissions = np.maximum(max_omissions, omissions)
        avg_omissions += omissions
    
    avg_omissions /= 33.0
    
    for i in range(len(df)):
        if i >= 6:
            recent_data = red_data[max(0, i-6):i].flatten()
            unique, counts = np.unique(recent_data, return_counts=True)
            hot_counts[i] = np.sum(counts >= 2)
            cold_counts[i] = 33 - len(unique)
    
    df['Hot_Count'] = hot_counts
    df['Cold_Count'] = cold_counts
    df['Max_Omission'] = max_omissions
    df['Avg_Omission'] = avg_omissions
    
    return df


def calc_poisson_features(df, window_size=100):
    """计算每个号码在滑动窗口内的泊松分布概率 (k=1)（向量化版）

    Args:
        df: 原始数据 DataFrame
        window_size: 滑动窗口大小

    Returns:
        pd.DataFrame: 包含 Poisson_R1~33 和 Poisson_B1~16 列的 DataFrame
    """
    red_data = df[TARGET_COLS[:6]].values
    blue_data = df['Blue1'].values
    
    poisson_data = np.zeros((len(df), RED_NUMBERS + BLUE_NUMBERS), dtype=np.float32)
    
    for num in range(1, RED_NUMBERS + 1):
        mask = (red_data == num).astype(np.float32).sum(axis=1)
        padded = np.zeros(len(df) + window_size - 1, dtype=np.float32)
        padded[window_size - 1:] = mask
        conv = np.convolve(padded, np.ones(window_size, dtype=np.float32), mode='valid')
        lam = conv / window_size
        poisson_data[:, num - 1] = poisson.pmf(1, lam)
    
    for num in range(1, BLUE_NUMBERS + 1):
        mask = (blue_data == num).astype(np.float32)
        padded = np.zeros(len(df) + window_size - 1, dtype=np.float32)
        padded[window_size - 1:] = mask
        conv = np.convolve(padded, np.ones(window_size, dtype=np.float32), mode='valid')
        lam = conv / window_size
        poisson_data[:, RED_NUMBERS + num - 1] = poisson.pmf(1, lam)
    
    columns = [f'Poisson_R{i}' for i in range(1, RED_NUMBERS + 1)] + [f'Poisson_B{i}' for i in range(1, BLUE_NUMBERS + 1)]
    return pd.DataFrame(poisson_data, columns=columns)


def calc_normal_params(df):
    """计算和值的正态分布参数（防守端）

    Args:
        df: 包含 Sum 列的 DataFrame

    Returns:
        pd.DataFrame: 添加了 Norm_Mean, Norm_Std 列的 DataFrame
    """
    df = df.copy()
    sum_mean = df['Sum'].mean()
    sum_std = df['Sum'].std()
    df['Norm_Mean'] = sum_mean
    df['Norm_Std'] = sum_std
    return df


def calc_entropy_features(df):
    """计算近期号码分布的信息熵（裁判端）（向量化版）

    Args:
        df: 原始数据 DataFrame

    Returns:
        pd.DataFrame: 添加了 Entropy 列的 DataFrame
    """
    df = df.copy()
    red_data = df[TARGET_COLS[:6]].values
    entropy_vals = np.zeros(len(df), dtype=np.float32)
    
    for i in range(ENTROPY_WINDOW, len(df)):
        window_data = red_data[i - ENTROPY_WINDOW:i].flatten()
        unique, counts = np.unique(window_data, return_counts=True)
        probs = counts / len(window_data)
        entropy_vals[i] = -np.sum(probs * np.log2(probs))
    
    df['Entropy'] = entropy_vals
    return df


def calc_markov_features(df):
    """计算奇偶比和大小比的马尔可夫链转移概率（进攻端）

    Args:
        df: 包含 OddRatio, BigRatio 列的 DataFrame

    Returns:
        pd.DataFrame: 添加了 Markov_Odd_Prob_0~6 和 Markov_Big_Prob_0~6 列的 DataFrame
    """
    df = df.copy()
    
    # 初始化转移矩阵
    markov_matrix_odd = np.zeros((MARKOV_STATES, MARKOV_STATES))
    markov_matrix_big = np.zeros((MARKOV_STATES, MARKOV_STATES))
    
    # 统计转移次数
    for i in range(1, len(df)):
        prev_odd_state = int(df.loc[i-1, 'OddRatio'] * 6)
        curr_odd_state = int(df.loc[i, 'OddRatio'] * 6)
        prev_big_state = int(df.loc[i-1, 'BigRatio'] * 6)
        curr_big_state = int(df.loc[i, 'BigRatio'] * 6)
        
        markov_matrix_odd[prev_odd_state, curr_odd_state] += 1
        markov_matrix_big[prev_big_state, curr_big_state] += 1
    
    # 归一化为概率
    with np.errstate(divide='ignore', invalid='ignore'):
        markov_prob_odd = markov_matrix_odd / markov_matrix_odd.sum(axis=1, keepdims=True)
        markov_prob_big = markov_matrix_big / markov_matrix_big.sum(axis=1, keepdims=True)
        markov_prob_odd = np.nan_to_num(markov_prob_odd)
        markov_prob_big = np.nan_to_num(markov_prob_big)
    
    # 生成每行的转移概率特征
    next_odd_probs = []
    next_big_probs = []
    for i in range(len(df)):
        curr_odd = int(df.loc[i, 'OddRatio'] * 6)
        curr_big = int(df.loc[i, 'BigRatio'] * 6)
        next_odd_probs.append(markov_prob_odd[curr_odd])
        next_big_probs.append(markov_prob_big[curr_big])
    
    df = pd.concat([
        df, 
        pd.DataFrame(next_odd_probs, columns=[f'Markov_Odd_Prob_{i}' for i in range(MARKOV_STATES)])
    ], axis=1)
    
    df = pd.concat([
        df, 
        pd.DataFrame(next_big_probs, columns=[f'Markov_Big_Prob_{i}' for i in range(MARKOV_STATES)])
    ], axis=1)
    
    return df


def generate_regression_targets(df):
    """生成下一期的回归目标特征

    Args:
        df: 包含统计特征列的 DataFrame

    Returns:
        pd.DataFrame: 添加了 Next_Sum, Next_OddRatio 等回归目标列的 DataFrame
    """
    df = df.copy()
    df['Next_Sum'] = df['Sum'].shift(-1)
    df['Next_OddRatio'] = df['OddRatio'].shift(-1)
    df['Next_BigRatio'] = df['BigRatio'].shift(-1)
    df['Next_Hot'] = df['Hot_Count'].shift(-1)
    df['Next_Cold'] = df['Cold_Count'].shift(-1)
    df['Next_Max_Omission'] = df['Max_Omission'].shift(-1)
    df['Next_Avg_Omission'] = df['Avg_Omission'].shift(-1)
    return df


def calculate_derived_features(df, window_size):
    """计算衍生特征（集成三大数学模型）的主入口

    调用顺序：基础特征 -> One-Hot编码 -> 统计特征 -> 泊松特征 -> 
              正态分布参数 -> 信息熵 -> 马尔可夫链 -> 回归目标

    Args:
        df: 原始数据 DataFrame
        window_size: 滑动窗口大小

    Returns:
        pd.DataFrame: 包含所有衍生特征的数据，已去除NaN行
    """
    start_time = time.time()
    df = df.copy()
    
    t_start = time.time()
    df = calc_base_features(df)
    elapsed = time.time() - t_start
    print(f"  [耗时 {elapsed:.2f}s] 基础特征")
    
    t_start = time.time()
    df = calc_onehot_features(df)
    elapsed = time.time() - t_start
    print(f"  [耗时 {elapsed:.2f}s] One-Hot编码特征")
    
    t_start = time.time()
    df = calc_statistical_features(df, window_size)
    elapsed = time.time() - t_start
    print(f"  [耗时 {elapsed:.2f}s] 统计特征")
    
    t_start = time.time()
    poisson_df = calc_poisson_features(df, window_size=POISSON_WINDOW)
    df = df.reset_index(drop=True)
    df = pd.concat([df, poisson_df.reset_index(drop=True)], axis=1)
    elapsed = time.time() - t_start
    print(f"  [耗时 {elapsed:.2f}s] 泊松分布特征")
    
    t_start = time.time()
    df = calc_normal_params(df)
    elapsed = time.time() - t_start
    print(f"  [耗时 {elapsed:.2f}s] 正态分布参数")
    
    t_start = time.time()
    df = calc_entropy_features(df)
    elapsed = time.time() - t_start
    print(f"  [耗时 {elapsed:.2f}s] 信息熵特征")
    
    t_start = time.time()
    df = calc_markov_features(df)
    elapsed = time.time() - t_start
    print(f"  [耗时 {elapsed:.2f}s] 马尔可夫链特征")
    
    t_start = time.time()
    df = generate_regression_targets(df)
    elapsed = time.time() - t_start
    print(f"  [耗时 {elapsed:.2f}s] 回归目标特征")
    
    total_time = time.time() - start_time
    print(f"\n特征工程总耗时: {total_time:.2f}秒")
    print(f"有效记录数: {len(df)}")
    
    return df.dropna().reset_index(drop=True)


# ================= 数据集与模型模块 =================
class HybridDataset(Dataset):
    """混合任务数据集，同时包含分类和回归目标"""
    
    def __init__(self, X, y_class, y_reg):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y_class = torch.tensor(y_class, dtype=torch.long)
        self.y_reg = torch.tensor(y_reg, dtype=torch.float32)
    
    def __len__(self):
        return len(self.X)
    
    def __getitem__(self, idx):
        return self.X[idx].permute(1, 0), self.y_class[idx], self.y_reg[idx]


class HybridMultiTaskCNN(nn.Module):
    """混合多任务 CNN 模型，同时进行分类和回归预测（带正则化）"""
    
    def __init__(self, num_channels):
        super(HybridMultiTaskCNN, self).__init__()
        self.conv1 = nn.Conv1d(in_channels=num_channels, out_channels=CNN_OUT_CHANNELS, kernel_size=CNN_KERNEL_SIZE, padding=CNN_KERNEL_SIZE//2)
        self.bn1 = nn.BatchNorm1d(CNN_OUT_CHANNELS)
        self.pool = nn.MaxPool1d(kernel_size=CNN_POOL_SIZE, stride=CNN_POOL_SIZE)
        self.relu = nn.ReLU()
        self.dropout1 = nn.Dropout(DROPOUT_RATE)
        self.fc1 = None  # 懒初始化
        self.dropout2 = nn.Dropout(DROPOUT_RATE)
        self.cls_head = nn.Linear(FC_HIDDEN_SIZE, (6 * RED_STATES) + (1 * BLUE_STATES))
        self.reg_head = nn.Linear(FC_HIDDEN_SIZE, len(REGRESSION_TARGETS))
        
    def forward(self, x):
        """前向传播
        
        Args:
            x: 输入张量，形状 (batch_size, num_channels, window_length)
        
        Returns:
            tuple: (cls_out, reg_out)
        """
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.pool(x)
        x = self.dropout1(x)
        
        x = x.view(x.size(0), -1)
        
        # 懒初始化全连接层
        if self.fc1 is None or self.fc1.in_features != x.size(1):
            self.fc1 = nn.Linear(x.size(1), FC_HIDDEN_SIZE).to(x.device)
        
        features = self.relu(self.fc1(x))
        features = self.dropout2(features)
        
        return self.cls_head(features), self.reg_head(features)


# ================= 训练与损失计算模块 =================
def compute_joint_loss(cls_out, reg_out, labels_cls, labels_reg):
    """计算多任务联合损失
    
    loss = cls_loss(6红球+1蓝球) + 0.5 * reg_loss(7个统计特征)

    Args:
        cls_out: 分类输出，形状 (batch_size, 6*RED_STATES + BLUE_STATES)
        reg_out: 回归输出，形状 (batch_size, num_regression_targets)
        labels_cls: 分类标签，形状 (batch_size, 7)
        labels_reg: 回归标签，形状 (batch_size, num_regression_targets)

    Returns:
        tuple: (total_loss, cls_loss, reg_loss)
    """
    cls_criterion = nn.CrossEntropyLoss()
    reg_criterion = nn.MSELoss()
    
    cls_loss = 0
    current_idx = 0
    for i in range(6):
        cls_loss += cls_criterion(cls_out[:, current_idx:current_idx + RED_STATES], labels_cls[:, i])
        current_idx += RED_STATES
    cls_loss += cls_criterion(cls_out[:, current_idx:current_idx + BLUE_STATES], labels_cls[:, 6])
    
    reg_loss = reg_criterion(reg_out, labels_reg)
    total_loss = cls_loss + REGRESSION_LOSS_WEIGHT * reg_loss
    
    return total_loss, cls_loss, reg_loss


def train_hybrid_model(X, y_class, y_reg, num_channels):
    """训练混合多任务 CNN 模型

    Args:
        X: 特征窗口数组
        y_class: 分类目标数组
        y_reg: 回归目标数组
        num_channels: 输入通道数

    Returns:
        tuple: (model, device)
    """
    t_start = time.time()
    X_train, X_temp, y_cls_train, y_cls_temp, y_reg_train, y_reg_temp = train_test_split(
        X, y_class, y_reg, test_size=TEST_SIZE, random_state=42
    )
    X_val, X_test, y_cls_val, y_cls_test, y_reg_val, y_reg_test = train_test_split(
        X_temp, y_cls_temp, y_reg_temp, test_size=0.5, random_state=42
    )
    
    train_loader = DataLoader(HybridDataset(X_train, y_cls_train, y_reg_train), batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(HybridDataset(X_val, y_cls_val, y_reg_val), batch_size=BATCH_SIZE, shuffle=False)
    
    elapsed = time.time() - t_start
    print(f"训练数据准备完成, 耗时: {elapsed:.2f}秒")
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = HybridMultiTaskCNN(num_channels=num_channels).to(device)
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', factor=LR_SCHEDULER_FACTOR, patience=LR_SCHEDULER_PATIENCE)
    
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    total_train_start = time.time()
    
    print(f"\n{'=' * 60}")
    print(f"开始训练模型 (设备: {device})")
    print(f"训练集: {len(train_loader.dataset)} 样本, 验证集: {len(val_loader.dataset)} 样本")
    print(f"批次大小: {BATCH_SIZE}, 输入通道数: {num_channels}")
    print(f"总参数: {total_params}, 可训练参数: {trainable_params}")
    print(f"模型结构: Conv1d({num_channels}→{CNN_OUT_CHANNELS}, kernel={CNN_KERNEL_SIZE}) → FC({FC_HIDDEN_SIZE}→{6*RED_STATES+BLUE_STATES})")
    print(f"{'=' * 60}")
    
    best_val_loss = float('inf')
    patience_counter = 0
    
    for epoch in range(EPOCHS):
        epoch_start = time.time()
        model.train()
        train_total_loss = 0
        train_cls_loss = 0
        train_reg_loss = 0
        
        for inputs, labels_cls, labels_reg in train_loader:
            inputs, labels_cls, labels_reg = inputs.to(device), labels_cls.to(device), labels_reg.to(device)
            optimizer.zero_grad()
            
            cls_out, reg_out = model(inputs)
            loss, cls_loss, reg_loss = compute_joint_loss(cls_out, reg_out, labels_cls, labels_reg)
            
            loss.backward()
            optimizer.step()
            
            train_total_loss += loss.item() * inputs.size(0)
            train_cls_loss += cls_loss.item() * inputs.size(0)
            train_reg_loss += reg_loss.item() * inputs.size(0)
        
        train_total_loss /= len(train_loader.dataset)
        train_cls_loss /= len(train_loader.dataset)
        train_reg_loss /= len(train_loader.dataset)
        
        epoch_time = time.time() - epoch_start
        
        if (epoch + 1) % TRAIN_LOG_INTERVAL == 0:
            val_start = time.time()
            model.eval()
            val_total_loss = 0
            val_cls_loss = 0
            val_reg_loss = 0
            
            with torch.no_grad():
                for inputs, labels_cls, labels_reg in val_loader:
                    inputs, labels_cls, labels_reg = inputs.to(device), labels_cls.to(device), labels_reg.to(device)
                    cls_out, reg_out = model(inputs)
                    loss, cls_loss, reg_loss = compute_joint_loss(cls_out, reg_out, labels_cls, labels_reg)
                    
                    val_total_loss += loss.item() * inputs.size(0)
                    val_cls_loss += cls_loss.item() * inputs.size(0)
                    val_reg_loss += reg_loss.item() * inputs.size(0)
            
            val_total_loss /= len(val_loader.dataset)
            val_cls_loss /= len(val_loader.dataset)
            val_reg_loss /= len(val_loader.dataset)
            
            scheduler.step(val_total_loss)
            val_time = time.time() - val_start
            
            current_lr = optimizer.param_groups[0]['lr']
            
            print(f"\nEpoch [{epoch+1}/{EPOCHS}]")
            print(f"  训练损失: {train_total_loss:.6f}, 训练时间: {epoch_time:.1f}s")
            print(f"  验证损失: {val_total_loss:.6f}, 验证时间: {val_time:.1f}s")
            print(f"  当前学习率: {current_lr:.6f}")
            
            if val_total_loss < best_val_loss:
                best_val_loss = val_total_loss
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
    
    return model, device


# ================= 数据准备模块 =================
def prepare_hybrid_data(df, window_size, step):
    """准备混合任务的训练数据，构造滑动窗口

    Args:
        df: 包含所有特征的 DataFrame
        window_size: 滑动窗口大小
        step: 滑动步长

    Returns:
        tuple: (X, y_class, y_reg) 或 (None, None, None)
    """
    full_data = extract_feature_matrix(df)
    reg_data = df[REGRESSION_TARGETS].values.astype(np.float32)
    
    n = full_data.shape[0]
    num_windows = (n - window_size) // step + 1
    if num_windows <= 0:
        return None, None, None
    
    X, y_class, y_reg = [], [], []
    for i in range(num_windows):
        X.append(full_data[i:i + window_size - 1])
        
        next_draw = full_data[i + window_size - 1, :7].copy()
        for j in range(6):
            next_draw[j] -= (j + 1)
        next_draw[6] -= 1
        y_class.append(next_draw)
        
        y_reg.append(reg_data[i + window_size - 1])
    
    return np.array(X), np.array(y_class), np.array(y_reg)


# ================= 预测后处理模块 =================
def check_entropy_chaos(df):
    """检查信息熵是否处于高混乱状态（裁判端）

    Args:
        df: 包含 Entropy 列的 DataFrame

    Returns:
        bool: 是否处于混乱状态
    """
    current_entropy = df.iloc[-1]['Entropy']
    is_chaos = current_entropy > ENTROPY_CHAOS_THRESHOLD
    if is_chaos:
        print(f"裁判端警告：信息熵 {current_entropy:.4f} > {ENTROPY_CHAOS_THRESHOLD}，系统高混乱状态！")
    return is_chaos


def extract_preliminary_preds(cls_out, is_chaos):
    """从模型输出中提取初步预测号码

    Args:
        cls_out: 分类输出，形状 (6*RED_STATES + BLUE_STATES,)
        is_chaos: 是否处于混乱状态

    Returns:
        tuple: (red_preds, raw_blue_pred)
    """
    red_preds = []
    current_idx = 0
    
    for i in range(6):
        red_logits = cls_out[current_idx:current_idx + RED_STATES]
        if is_chaos:
            red_logits = red_logits * CHAOS_DAMPING_FACTOR + np.mean(red_logits) * (1 - CHAOS_DAMPING_FACTOR)
        
        pred_idx = np.argmax(red_logits)
        red_preds.append(pred_idx + (i + 1))
        current_idx += RED_STATES
    
    blue_logits = cls_out[current_idx:current_idx + BLUE_STATES]
    if is_chaos:
        blue_logits = blue_logits * CHAOS_DAMPING_FACTOR + np.mean(blue_logits) * (1 - CHAOS_DAMPING_FACTOR)
    
    raw_blue_pred = np.argmax(blue_logits) + 1
    return red_preds, raw_blue_pred


def apply_sum_constraint(red_preds, predicted_sum):
    """根据回归预测的和值约束红球和值范围

    Args:
        red_preds: 当前红球预测列表
        predicted_sum: 回归预测的下一期红球和值

    Returns:
        list: 修正后的红球预测列表
    """
    current_sum = sum(red_preds)
    diff = predicted_sum - current_sum
     
    if abs(diff) <= SUM_CONSTRAINT_THRESHOLD:
        return red_preds
    
    print(f"和值修正: 当前和值 {current_sum}, 预测和值 {predicted_sum:.1f}, 差值: {diff:.1f}")
    
    if diff > 0:
        max_red = max(red_preds)
        if max_red < 33:
            for i in range(max_red + 1, 34):
                if i not in red_preds:
                    red_preds.remove(max_red)
                    red_preds.append(i)
                    break
    else:
        min_red = min(red_preds)
        if min_red > 1:
            for i in range(min_red - 1, 0, -1):
                if i not in red_preds:
                    red_preds.remove(min_red)
                    red_preds.append(i)
                    break
    
    red_preds.sort()
    print(f"修正后红球: {red_preds}, 新和值: {sum(red_preds)}")
    return red_preds


def apply_normal_filter(red_preds, df):
    """应用正态分布过滤，修正和值异常的预测（防守端）

    Args:
        red_preds: 初步红球预测列表
        df: 包含 Norm_Mean, Norm_Std 列的 DataFrame

    Returns:
        list: 修正后的红球预测列表
    """
    last_row = df.iloc[-1]
    sum_mean = last_row['Norm_Mean']
    sum_std = last_row['Norm_Std']
    
    pred_sum = sum(red_preds)
    lower_bound = sum_mean - (NORMAL_SIGMA * sum_std)
    upper_bound = sum_mean + (NORMAL_SIGMA * sum_std)
    
    if pred_sum < lower_bound or pred_sum > upper_bound:
        print(f"防守端警告：和值 {pred_sum} 超出范围 [{lower_bound:.1f}, {upper_bound:.1f}]，正在修正...")
        
        if pred_sum > upper_bound:
            # 和值过大，替换最大的号码
            max_red = max(red_preds)
            for i in range(max_red - 1, 0, -1):
                if i not in red_preds:
                    red_preds.remove(max_red)
                    red_preds.append(i)
                    break
        else:
            # 和值过小，替换最小的号码
            min_red = min(red_preds)
            for i in range(min_red + 1, 34):
                if i not in red_preds:
                    red_preds.remove(min_red)
                    red_preds.append(i)
                    break
        
        red_preds.sort()
        print(f"修正后红球: {red_preds}, 新和值: {sum(red_preds)}")
    
    return red_preds


def apply_poisson_optimization(red_preds, raw_blue_pred, df):
    """根据泊松概率优化最终号码选择

    Args:
        red_preds: 红球预测列表
        raw_blue_pred: 初步蓝球预测
        df: 包含 Poisson_R1~33 和 Poisson_B1~16 列的 DataFrame

    Returns:
        tuple: (final_reds, final_blue)
    """
    last_row = df.iloc[-1]
    poisson_red_probs = last_row[[f'Poisson_R{i}' for i in range(1, 34)]].values
    poisson_blue_probs = last_row[[f'Poisson_B{i}' for i in range(1, 17)]].values
    
    # 红球优化
    candidates = []
    for r in red_preds:
        candidates.append(r if poisson_red_probs[r - 1] > 0 else -1)
    
    sorted_indices = np.argsort(poisson_red_probs)[::-1]
    for i in sorted_indices:
        num = i + 1
        if len(candidates) >= 6:
            break
        if num not in candidates:
            if -1 in candidates:
                candidates[candidates.index(-1)] = num
            else:
                candidates.append(num)
    
    final_reds = sorted(candidates[:6])
    
    # 蓝球优化
    final_blue = raw_blue_pred
    if poisson_blue_probs[raw_blue_pred - 1] == 0:
        final_blue = np.argmax(poisson_blue_probs) + 1
    
    return final_reds, final_blue


def predict_next_hybrid(model, df, device):
    """输入最新数据，预测下一期开奖结果

    预测流程：构造输入 -> 模型推理 -> 信息熵检查 -> 初步预测 -> 正态过滤 -> 泊松优化

    Args:
        model: 训练好的模型
        df: 包含所有特征的 DataFrame
        device: 模型运行设备

    Returns:
        tuple: (predictions, reg_features)
    """
    # 构造输入数据
    recent_window = df.tail(WINDOW_SIZE - 1)
    full_data = extract_feature_matrix(recent_window)
    input_tensor = torch.tensor(full_data, dtype=torch.float32).permute(1, 0).unsqueeze(0).to(device)
    
    # 模型推理
    model.eval()
    with torch.no_grad():
        cls_out, reg_out = model(input_tensor)
        cls_out = cls_out.cpu().numpy()[0]
        reg_out = reg_out.cpu().numpy()[0]
    
    # 信息熵检查（裁判端）
    is_chaos = check_entropy_chaos(df)
    
    # 提取初步预测
    red_preds, raw_blue_pred = extract_preliminary_preds(cls_out, is_chaos)
    
    # 正态分布过滤（防守端）
    red_preds = apply_normal_filter(red_preds, df)
    
    # 泊松优化
    final_reds, final_blue = apply_poisson_optimization(red_preds, raw_blue_pred, df)
    
    # 回归和值修正
    final_reds = apply_sum_constraint(final_reds, reg_out[0])
    
    predictions = final_reds + [final_blue]
    reg_features = {name: round(val, 4) for name, val in zip(REGRESSION_TARGETS, reg_out)}
    
    return np.array(predictions), reg_features


# ================= 模型保存与加载模块 =================
def save_model(model, path):
    """保存模型权重到文件"""
    torch.save(model.state_dict(), path)
    print(f"模型已保存至: {path}")


def load_model(model, path, device):
    """从文件加载模型权重"""
    model.load_state_dict(torch.load(path, map_location=device))
    print(f"模型已从 {path} 加载")
    return model


# ================= 主程序入口 =================
if __name__ == "__main__":
    total_start = time.time()
    
    print("=" * 60)
    print(f"CNN 双色球(红球+蓝球)概率预测系统 - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 60)
    
    # 1. 加载数据
    t_start = time.time()
    data_file = Path("C:/Users/lw25622/ML/SSQ/1.csv")
    
    try:
        df = pd.read_csv(data_file)
        elapsed = time.time() - t_start
        print(f"\n正在加载数据: {data_file}")
        print(f"✓ 数据加载成功, 共 {len(df)} 期记录, 耗时: {elapsed:.2f}秒")
    except FileNotFoundError:
        print(f"\n✗ 未找到数据文件: {data_file}")
        exit()
    
    # 2. 计算衍生特征
    print("\n" + "=" * 60)
    print("开始特征工程...")
    df = calculate_derived_features(df, WINDOW_SIZE)
    
    # 3. 计算输入通道数
    INPUT_CHANNELS = 7 + 33 + 49 + 2 + 1 + (MARKOV_STATES * 2)
    print(f"\n输入通道数: {INPUT_CHANNELS}")
    
    # 4. 准备训练数据
    t_start = time.time()
    X, y_class, y_reg = prepare_hybrid_data(df, WINDOW_SIZE, WINDOW_STEP)
    if X is not None:
        elapsed = time.time() - t_start
        print(f"滑动窗口生成完成 - X: {X.shape}, y_class: {y_class.shape}, y_reg: {y_reg.shape}, 耗时: {elapsed:.2f}秒")
    else:
        print("数据量不足以生成窗口！")
        exit()
    
    # 5. 训练或加载模型
    model_path = "hybrid_math_model.pth"
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = HybridMultiTaskCNN(num_channels=INPUT_CHANNELS).to(device)
    
    if RETRAIN_MODEL == 'Y':
        print("\n" + "=" * 60)
        print("开始训练模型...")
        model, device = train_hybrid_model(X, y_class, y_reg, INPUT_CHANNELS)
        save_model(model, model_path)
    else:
        print("\n" + "=" * 60)
        print("加载已有模型...")
        try:
            model = load_model(model, model_path, device)
        except FileNotFoundError:
            print("模型文件不存在，自动训练...")
            model, device = train_hybrid_model(X, y_class, y_reg, INPUT_CHANNELS)
            save_model(model, model_path)
    
    # 6. 预测下一期
    t_start = time.time()
    print("\n" + "=" * 60)
    print("开始预测下一期开奖结果...")
    preds, reg_preds = predict_next_hybrid(model, df, device)
    elapsed = time.time() - t_start
    print(f"预测完成, 耗时: {elapsed:.2f}秒")
    
    print(f"\n预测结果 (基于过去{WINDOW_SIZE-1}期):")
    print("-" * 64)
    print(f"红球预测: {preds[:6]}")
    print(f"蓝球预测: {preds[6]}")
    print(f"{'-'*64}")
    print("预测的统计特征:")
    for k, v in reg_preds.items():
        print(f"  - {k}: {v}")
    print(f"{'-'*64}")
    
    # 7. 总耗时统计
    total_time = time.time() - total_start
    print(f"\n{'=' * 60}")
    print(f"CNN6-1 处理完成!")
    print(f"总耗时: {total_time:.2f}秒")
    print(f"{'=' * 60}")
