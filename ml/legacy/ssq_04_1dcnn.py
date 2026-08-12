import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import LabelEncoder
from pathlib import Path
import time
import json
from datetime import datetime

# 1. 定义一个全局列表，用于收集所有的预测 JSON 数据
predictions_list = []
data_list = []

WINDOW_SIZE = 100
WINDOW_STEP = 1
TOP_PRO=6

# ================= 1. 工具函数 =================
def sliding_window_numpy(data_list, window_size, step):
    """使用 NumPy 内存视图实现高性能滑动窗口拆解"""
    arr = np.asarray(data_list)
    n = arr.shape[0]
    num_windows = (n - window_size) // step + 1
    strides = arr.strides[0]
    new_shape = (num_windows, window_size)
    new_strides = (strides * step, strides)
    windows = np.lib.stride_tricks.as_strided(arr, shape=new_shape, strides=new_strides)
    return pd.DataFrame(windows)

def top3_accuracy(y_true, y_pred_proba):
    """计算 Top-3 命中率"""
    correct = 0
    for true_label, proba in zip(y_true, y_pred_proba):
        top_3_indices = np.argsort(proba)[-TOP_PRO:]
        if true_label in top_3_indices:
            correct += 1
    return correct / len(y_true)

# ================= 2. PyTorch 专属组件 =================
class TimeSeriesDataset(Dataset):
    """自定义数据集,用于将 Pandas DataFrame 转换为 PyTorch Tensor
    dtype=torch.long:PyTorch 规定,分类任务的标签必须是整数类型(Long)。
    unsqueeze(0):这是 CNN 初学者最容易踩坑的地方！ 1D-CNN 要求输入必须是三维的:(Batch_size, Channels, Length)。
    你的数据原本是二维的 (Batch, Length),这里强行在最前面加了一个维度,变成 (Batch, 1, Length),代表单通道(Channel=1)。
    """
    def __init__(self, X, y):
        self.X = torch.tensor(X.values, dtype=torch.float32)    # 特征转浮点数
        # 核心安全点:将 LabelEncoder 编码后的 0-based 整数转为 LongTensor
        self.y = torch.tensor(y, dtype=torch.long)              # 标签转长整型

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        # CNN 需要 3D 输入: (Batch, Channels, Length) -> 这里 Channel=1
        return self.X[idx].unsqueeze(0), self.y[idx]        # 核心:增加一个维度

class CNN1DModel(nn.Module):
    """
        1D-CNN 模型定义
        ① 卷积层 nn.Conv1d
            in_channels=1:输入通道数为 1(因为我们只有一个维度的数字序列)。
            out_channels=64:输出通道数(即卷积核的数量)。
            作用:相当于同时派出 64 个不同的“特征探测器”去扫描数据。
            影响:设置越大,模型能学到的特征越丰富,但参数变多,容易过拟合；设置太小,模型学不到东西(欠拟合)。
            kernel_size=3:卷积核的大小(窗口宽度)。
            作用:每次只看相邻的 3 个数字,提取局部的 3 连号规律。
            影响:设为 3 是最常见的经验值。如果设为 5 或 7,能捕捉更长范围的规律,但会丢失更细微的局部特征。
            padding=1:边缘填充。
            作用:在序列两头各补 1 个 0。如果不补,331 的长度经过 kernel=3 的卷积后会变成 329。加了 padding=1 后,长度依然保持 331。这保证了特征图不会因为卷积而缩水。
        ② 池化层 nn.MaxPool1d
            kernel_size=2, stride=2:
            作用:把 331 个特征点,每 2 个取一个最大值,结果长度直接减半(变成 165)。
            影响:极大地减少了后续全连接层的计算量,同时让模型对微小的数字波动产生“钝感”(抗噪能力)。
        ③ 全连接层 nn.Linear
            flattened_size = 64 * (input_length // 2):
            作用:动态计算 Flatten 后的节点数。因为经过了池化,长度减半,通道数是 64,所以总特征数 = 64 * 165。
            fc1 = nn.Linear(flattened_size, 128):隐藏层,把高维特征压缩到 128 维。
            fc2 = nn.Linear(128, num_classes):输出层,输出类别数(比如红球有 33 个类别,这里就是 33)。
    """
    def __init__(self, input_length, num_classes):
        super(CNN1DModel, self).__init__()
        self.conv1 = nn.Conv1d(in_channels=1, out_channels=32, kernel_size=7, padding=1)
        self.pool = nn.MaxPool1d(kernel_size=2, stride=2)
        self.relu = nn.ReLU()
        
        # fc1 隐藏层：将高维特征压缩到 128 维（这里使用动态计算，避免报错）
        # 注意：在 __init__ 中先不定义 fc1，我们在 forward 中动态初始化
        self.fc1 = None  
        
        # 输出层：输入是 128，输出是类别数 (num_classes)
        self.fc2 = nn.Linear(128, num_classes)

    def forward(self, x):
        # 卷积 -> 激活 -> 池化
        x = self.pool(self.relu(self.conv1(x)))
        # 展平 (Flatten)：将二维特征图变成一维向量
        x = x.view(x.size(0), -1)  # Flatten
        # 动态获取展平后的真实特征数，并初始化 fc1
        real_flattened_size = x.size(1)
        if self.fc1 is None or self.fc1.in_features != real_flattened_size:
            self.fc1 = nn.Linear(real_flattened_size, 128).to(x.device)

        # 隐藏层 -> 激活
        x = self.relu(self.fc1(x))
        # 输出层 (输出原始 logits，不需要在这里加 Softmax)
        x = self.fc2(x)
        return x

# ================= 3. 核心处理流程 =================
def prepare_data(column_name, df, window_size, step):
    """子流程 1: 数据准备与编码"""
    print(f"[{column_name}] 正在生成滑动窗口特征...")
    subItem = sliding_window_numpy(df[column_name], window_size=window_size, step=step)
    lastColName = subItem.columns[-1]
    subItem = subItem.rename(columns={lastColName: 'label'})
    
    X = subItem.iloc[:, :-1]
    y_raw = subItem.iloc[:, -1]

    # 过滤掉出现次数少于 5 次的极端长尾标签
    value_counts = y_raw.value_counts()
    valid_labels = value_counts[value_counts >= 3].index
    valid_mask = y_raw.isin(valid_labels)
    
    X = X[valid_mask].reset_index(drop=True)
    y_raw = y_raw[valid_mask].reset_index(drop=True)
    
    print(f"[{column_name}] 正在初始化 LabelEncoder 并转换标签...")
    le = LabelEncoder()
    y = le.fit_transform(y_raw)
    num_classes = len(le.classes_)
    print(f"[{column_name}] 编码完成。原始类别: {list(le.classes_)} -> 编码后类别数: {num_classes}")
    
    return X, y, le, num_classes

def train_cnn_model(X, y, num_classes, column_name):
    """子流程 2:1D-CNN 模型训练与评估
    CrossEntropyLoss:多分类任务的标配。它内部自动包含了 Softmax 操作,所以你的模型最后一层 fc2 不需要加 Softmax。
    Adam 优化器:目前深度学习最主流的优化器,自适应调整学习率,比传统的 SGD 收敛快得多。
    lr=0.001:学习率。决定了模型每次更新权重的步长。太大模型会震荡不收敛,太小模型学得极慢。
    model.train() 与 model.eval():PyTorch 的强制规范。训练时开启 Dropout 等随机机制；预测/评估时必须关闭,保证结果确定性。
    """
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    train_dataset = TimeSeriesDataset(X_train, y_train)
    test_dataset = TimeSeriesDataset(X_test, y_test)
    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)
    
    print(f"[{column_name}] 正在训练 1D-CNN 模型...")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = CNN1DModel(input_length=X.shape[1], num_classes=num_classes).to(device)
    
    criterion = nn.CrossEntropyLoss()  # 内部自带 Softmax,完美适配多分类
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    
    # 训练循环 (这里简单演示 20 个 Epoch,可根据需要增加)
    for epoch in range(61):
        model.train()
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
    # 评估阶段
    model.eval()
    all_preds_proba = []
    all_labels = []
    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs = inputs.to(device)
            outputs = model(inputs)
            probs = torch.softmax(outputs, dim=1).cpu().numpy()
            all_preds_proba.extend(probs)
            all_labels.extend(labels.numpy())
            
    all_preds_proba = np.array(all_preds_proba)
    all_labels = np.array(all_labels)
    
    top1_acc = accuracy_score(all_labels, np.argmax(all_preds_proba, axis=1))
    top3_acc = top3_accuracy(all_labels, all_preds_proba)
    print(f"[{column_name}] CNN 模型评估 -> Top-1 准确率: {top1_acc:.4f} | Top-3 命中率: {top3_acc:.4f}")
    
    return model, X_train, device, num_classes

def predict_and_evaluate(model, le, X_train_columns, df, column_name, device, num_classes):
    """子流程 3:新数据预测与结果解码"""
    print(f"[{column_name}] 正在预测最新数据...")
    new_data = df[column_name].tail(WINDOW_SIZE - 1).tolist()
    new_data_df = pd.DataFrame([new_data], columns=X_train_columns)
    
    # 转换为 Tensor 并调整维度 (1, 1, 33)
    input_tensor = torch.tensor(new_data_df.values, dtype=torch.float32).unsqueeze(0).to(device)
    
    model.eval()
    with torch.no_grad():
        output = model(input_tensor)
        probs = torch.softmax(output, dim=1).cpu().numpy()[0]
        
    top_3_indices = np.argsort(probs)[-TOP_PRO:]
    top_3_original_numbers = le.inverse_transform(top_3_indices)
    top_3_probs = probs[top_3_indices]

    # 2. 组装成指定的 JSON 结构
    prediction_json = {
        "predict_time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "target_column": column_name,
        "top_3_predictions": [
            {"number": int(num), "probability": round(float(prob), 4)} 
            for num, prob in zip(top_3_original_numbers, top_3_probs)
        ]
    }

    data_json = [
        {str(int(num)): round(float(prob), 4)} 
        for num, prob in zip(top_3_original_numbers, top_3_probs)
    ]

    # 3. 将 JSON 数据追加存入全局列表
    if column_name != "Blue1":
        predictions_list.append(prediction_json)
        data_list.extend(data_json)
    
    
    print(f"\n[{column_name}] 预测结果:")
    print("-" * 30)
    for number, prob in zip(top_3_original_numbers, top_3_probs):
        print(f"  最可能数字: {number}, 概率: {prob:.4f}")
    print("-" * 30)

# ================= 4. 主调度函数 =================
def process_column(column_name, df, window_size, step):
    """主流程:按顺序调用各个子流程,完成单列的完整生命周期。"""
    print(f"\n{'='*30} 开始处理: {column_name} {'='*30}")
    start_time = time.time()

    # 1. 准备数据
    X, y, le, num_classes = prepare_data(column_name, df, window_size, step)
    
    # 2. 训练 CNN
    model, X_train, device, _ = train_cnn_model(X, y, num_classes, column_name)
    
    # 3. 预测新数据
    predict_and_evaluate(model, le, X_train.columns, df, column_name, device, num_classes)
    
    elapsed_time = time.time() - start_time
    print(f"[{column_name}] 处理完成,总耗时: {elapsed_time:.2f} 秒")
    print(f"{'='*30} 结束处理: {column_name} {'='*30}")


# ================= 5. 主程序入口 =================
if __name__ == "__main__":
    home_path = Path(r"C:/Users/lw25622/ML")
    data_file = home_path / "data" / "ssq" / "1.csv"
    windowSize = WINDOW_SIZE
    windowStep = WINDOW_STEP
    
    print(f"{'='*30} 处理开始: {datetime.now()} {'='*30}")
    print(f"正在加载数据: {data_file}")
    df = pd.read_csv(data_file)
    print(f"数据加载成功，共 {len(df)} 条记录。\n")

    target_columns = ['Red1', 'Red2', 'Red3', 'Red4', 'Red5', 'Red6', 'Blue1']

    for col in target_columns:
        if col in df.columns:
            process_column(col, df, windowSize, windowStep)
        else:
            print(f"[警告] 数据集中未找到列: {col}，已跳过。")

    sorted_data2 = sorted(
        data_list,
        key=lambda x: int(list(x.keys())[0]),  # 提取第一个键（Key）
        reverse=False
    )
    from  datetime import datetime
    file_path = datetime.now().strftime("%Y%m%d")
    file_path = "./" + file_path + ".dat"

    df = pd.DataFrame.from_records(sorted_data2, index=None).T.stack().reset_index()
    df = df.drop(columns=['level_1'])
    df.columns = ['number', 'probability']

    # 3. 写入 CSV 文件
    df.to_csv(file_path, index=False, mode='a')
    print(f"数据已成功写入: {file_path} \n\n")
    