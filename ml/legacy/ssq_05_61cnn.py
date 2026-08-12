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
warnings.filterwarnings('ignore')

# ================= 全局参数配置 =================
WINDOW_SIZE = 99       # 使用过去99次记录作为输入
WINDOW_STEP = 1        # 滑动步长
BATCH_SIZE = 64        # 批次大小
EPOCHS = 50            # 训练轮数
LEARNING_RATE = 0.001  # 学习率
TARGET_COLS = ['Red1', 'Red2', 'Red3', 'Red4', 'Red5', 'Red6', 'Blue1']
RED_STATES = 28        # 每个红球位置的有效状态数
BLUE_STATES = 16       # 蓝球的有效状态数


# 【新增】：定义需要回归预测的连续特征
REGRESSION_TARGETS = [
    'Next_Sum', 
    'Next_OddRatio', 
    'Next_BigRatio', 
    'Next_Hot', 
    'Next_Cold', 
    'Next_Max_Omission', 
    'Next_Avg_Omission'
]

# ================= 1. 衍生特征计算引擎 =================
def calculate_derived_features(df, lookback_window=5):
    """
    计算历史衍生特征，并生成下一期的回归目标。
    参数:
        df (DataFrame): 原始数据集
        lookback_window (int): 回溯期数（用于计算冷热指标），默认为 5
    """
    df = df.copy()
    all_reds = df[TARGET_COLS[:6]]
    
    # 1. 和值
    df['Sum'] = all_reds.sum(axis=1)
    # 2. 奇偶比 (奇数个数 / 6)
    df['OddRatio'] = (all_reds % 2 == 1).sum(axis=1) / 6.0
    # 3. 大小比 (>=17为大盘，1~16为小盘，比例)
    df['BigRatio'] = (all_reds >= 17).sum(axis=1) / 6.0
    
    # 4. 【核心升级】：全局综合考虑冷热与遗漏
    df['Hot_Count'] = 0
    df['Cold_Count'] = 0
    df['Max_Omission'] = 0
    df['Avg_Omission'] = 0.0

    # 遍历需要的行
    # 注意：循环范围是 range(lookback_window, len(df))，所以 i+1 最大为 len(df)-1，不会越界
    for i in range(lookback_window, len(df)-1):
        # 【修正点1】：获取下一期的索引，因为我们是基于当前历史信息来预测下一期
        next_idx = df.index[i+1] 
        
        # 获取过去 lookback_window 期的所有红球数据
        past_draws = df[TARGET_COLS[:6]].iloc[i - lookback_window : i].values.flatten()
        
        # 获取当前期（第 i 期）之前的所有历史数据 (用于计算遗漏)
        history_draws_df = df[TARGET_COLS[:6]].iloc[:i]
        
        # 统计热号和冷号
        unique, counts = np.unique(past_draws, return_counts=True)
        hot_count = np.sum(counts >= 2)
        cold_count = 33 - len(unique)
        
        # 统计最大遗漏和平均遗漏
        omissions = []
        for num in range(1, 34):
            # 在历史数据 DataFrame 中查找号码 num 出现的所有位置
            rows, _ = np.where(history_draws_df.values == num)
            
            if len(rows) == 0:
                # 如果从未出现过，遗漏值设为一个较大的数
                omissions.append(next_idx)
            else:
                # 找到最后一次出现的行索引
                last_seen_row_in_history = rows[-1]
                # 获取该行的真实 DataFrame 索引
                last_seen_true_idx = history_draws_df.index[last_seen_row_in_history]
                # 【修正点2】：正确的遗漏值 = 下一期索引 - 上次出现期的索引
                omission_value = next_idx - last_seen_true_idx
                omissions.append(omission_value)

        max_omission = max(omissions)
        avg_omission = np.mean(omissions)
        
        # 赋值给当前行
        df.loc[df.index[i], 'Hot_Count'] = float(hot_count)
        df.loc[df.index[i], 'Cold_Count'] = float(cold_count)
        df.loc[df.index[i], 'Max_Omission'] = float(max_omission)
        df.loc[df.index[i], 'Avg_Omission'] = float(avg_omission)

    # 【关键】：生成下一期的目标（标签）
    # 将当前行的综合特征，作为预测下一期的输入
    df['Next_Sum'] = df['Sum'].shift(-1)
    df['Next_OddRatio'] = df['OddRatio'].shift(-1)
    df['Next_BigRatio'] = df['BigRatio'].shift(-1)
    df['Next_Hot'] = df['Hot_Count'].shift(-1)
    df['Next_Cold'] = df['Cold_Count'].shift(-1)
    df['Next_Max_Omission'] = df['Max_Omission'].shift(-1)
    df['Next_Avg_Omission'] = df['Avg_Omission'].shift(-1)
    
    # 删除因 shift(-1) 产生的最后一行空值
    return df.dropna().reset_index(drop=True)

# ================= 2. 混合数据集 =================
class HybridDataset(Dataset):
    def __init__(self, X, y_class, y_reg):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y_class = torch.tensor(y_class, dtype=torch.long)
        self.y_reg = torch.tensor(y_reg, dtype=torch.float32)
        
    def __len__(self): return len(self.X)
    def __getitem__(self, idx):
        return self.X[idx].permute(1, 0), self.y_class[idx], self.y_reg[idx]

# ================= 3. 混合架构 CNN 模型 =================
class HybridMultiTaskCNN(nn.Module):
    def __init__(self, num_channels):
        super(HybridMultiTaskCNN, self).__init__()
        self.conv1 = nn.Conv1d(in_channels=num_channels, out_channels=64, kernel_size=7, padding=3)
        self.pool = nn.MaxPool1d(kernel_size=2, stride=2)
        self.relu = nn.ReLU()
        self.fc1 = None
        
        # 分类头：6*28 + 1*16 = 184
        self.cls_head = nn.Linear(128, (6 * RED_STATES) + (1 * BLUE_STATES))
        # 回归头：预测 3 个连续值 (和值, 奇偶比, 大小比)
        self.reg_head = nn.Linear(128, len(REGRESSION_TARGETS)) 

    def forward(self, x):
        x = self.pool(self.relu(self.conv1(x)))
        x = x.view(x.size(0), -1)
        real_flattened_size = x.size(1)
        if self.fc1 is None or self.fc1.in_features != real_flattened_size:
            self.fc1 = nn.Linear(real_flattened_size, 128).to(x.device)
            
        features = self.relu(self.fc1(x))
        cls_out = self.cls_head(features)  # 输出 184 维
        reg_out = self.reg_head(features)  # 输出 3 维
        return cls_out, reg_out

# ================= 4. 核心处理流程 =================
def prepare_hybrid_data(df, window_size, step):
    data = df[TARGET_COLS].values.astype(np.int32)
    reg_data = df[REGRESSION_TARGETS].values.astype(np.float32)
    n = data.shape[0]
    num_windows = (n - window_size) // step + 1
    if num_windows <= 0: return None, None, None
    
    X, y_class, y_reg = [], [], []
    for i in range(num_windows):
        X.append(data[i : i + window_size - 1])
        
        # 分类标签映射 (0~27)
        next_draw = data[i + window_size - 1].copy()
        for j in range(6): next_draw[j] -= (j + 1)
        next_draw[6] -= 1
        y_class.append(next_draw)
        
        # 回归标签
        y_reg.append(reg_data[i + window_size - 1])
        
    return np.array(X), np.array(y_class), np.array(y_reg)

def train_hybrid_model(X, y_class, y_reg):
    X_train, X_test, y_cls_train, y_cls_test, y_reg_train, y_reg_test = train_test_split(
        X, y_class, y_reg, test_size=0.2, random_state=42
    )
    train_loader = DataLoader(HybridDataset(X_train, y_cls_train, y_reg_train), batch_size=BATCH_SIZE, shuffle=True)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = HybridMultiTaskCNN(num_channels=len(TARGET_COLS)).to(device)
    
    cls_criterion = nn.CrossEntropyLoss()
    reg_criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    
    for epoch in range(EPOCHS):
        model.train()
        total_loss = 0
        for inputs, labels_cls, labels_reg in train_loader:
            inputs = inputs.to(device)
            labels_cls = labels_cls.to(device)
            labels_reg = labels_reg.to(device)
            
            optimizer.zero_grad()
            cls_out, reg_out = model(inputs)
            
            # 联合损失计算
            loss = 0
            current_idx = 0
            for i in range(6): 
                loss += cls_criterion(cls_out[:, current_idx : current_idx + RED_STATES], labels_cls[:, i])
                current_idx += RED_STATES
            loss += cls_criterion(cls_out[:, current_idx : current_idx + BLUE_STATES], labels_cls[:, 6])
            
            # 回归损失 (权重设为0.5，可根据需要调整)
            loss += 0.5 * reg_criterion(reg_out, labels_reg)
            
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            
    print(f"  混合模型训练完成，最终联合 Loss: {total_loss:.4f}")
    return model, device

# ================= 5. 混合模型预测函数 =================
def predict_next_hybrid(model, df, device):
    """输入最新99期，预测下一期的整体结果（包含号码和统计特征）"""
    recent_data = df[TARGET_COLS].tail(WINDOW_SIZE - 1).values.astype(np.int32)
    
    # 维度转换：(98, 7) -> (1, 7, 98) 以适配 Conv1d
    input_tensor = torch.tensor([recent_data], dtype=torch.float32).permute(0, 2, 1).to(device)
    
    model.eval()
    with torch.no_grad():
        cls_out, reg_out = model(input_tensor)
        cls_out = cls_out.cpu().numpy()[0]
        reg_out = reg_out.cpu().numpy()[0]
        
    predictions = []
    current_idx = 0
    
    # 【反向映射】：将 0~27 的相对索引转换回真实的绝对红球号码
    for i in range(6): 
        red_logits = cls_out[current_idx : current_idx + RED_STATES]
        pred_idx = np.argmax(red_logits)
        real_number = pred_idx + (i + 1)  # 加回偏移量 (1, 2, 3, 4, 5, 6)
        predictions.append(real_number)
        current_idx += RED_STATES
        
    # 预测蓝球
    blue_logits = cls_out[current_idx : current_idx + BLUE_STATES]
    pred_idx = np.argmax(blue_logits)
    real_blue = pred_idx + 1  
    predictions.append(real_blue)
    
    # 预测回归特征（和值、奇偶比等）
    reg_features = {name: round(val, 4) for name, val in zip(REGRESSION_TARGETS, reg_out)}
    
    return np.array(predictions), reg_features

# ================= 6. 主程序入口 =================
if __name__ == "__main__":
    work_path = Path.cwd()
    data_file = work_path / "1.csv"
    print(data_file)

    print(f"{'='*30} 处理开始: {datetime.now()} {'='*30}")
    print(f"正在加载数据并计算衍生特征...")
    df = pd.read_csv(data_file)
    df = calculate_derived_features(df)

    print(f"数据处理成功，共 {len(df)} 期有效记录。\n")

    print("正在生成混合滑动窗口...")
    X, y_class, y_reg = prepare_hybrid_data(df, WINDOW_SIZE, WINDOW_STEP)
    
    if X is not None:
        print("开始训练混合架构模型...")
        model, device = train_hybrid_model(X, y_class, y_reg)
        
        # 【新增】：调用混合预测函数并输出结果
        preds, reg_preds = predict_next_hybrid(model, df, device)
        
        print(f"\n🎯 整体预测结果 (基于过去99期):")
        print(f"红球: {preds[:6].tolist()} | 蓝球: {preds[6]}")
        print(f"预测统计特征: {reg_preds}")
    else:
        print("数据量不足以生成窗口！")