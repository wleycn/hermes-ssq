"""双色球预测系统 - Transformer 模型（全量输出: 红球33 + 蓝球16）

技术定位: 该项目已实证双色球为独立均匀随机过程，本模型仅用于技术栈完整性验证
（PyTorch TransformerEncoder 在序列建模上的能力展示），预期增益 ≈ 0。
因此必须强正则化（dropout 0.3~0.5 + weight_decay + 早停 + 小学习率）防止过拟合
（历史数据仅 3489 期，特征 302 列）。

结构与 lstm_all 对齐:
  - 输入: (batch, window_size, feat_dim) 窗口序列
  - 输出: 49 维 sigmoid = 红球 one-hot 累加(33) + 蓝球 one-hot(16)
  - 数据准备: StandardScaler 特征标准化 + 滑动窗口（仿 LSTMAllModel.prepare_data）
"""
import json
import time
import warnings
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from sklearn.preprocessing import StandardScaler

from ml.config import RED_COLS
from ml.models.base_model import BaseModel

warnings.filterwarnings("ignore")

# ================= Transformer 默认配置 =================
# 独立于 ml/config.py（任务要求不修改现有文件），默认值内聚在本模块。
TRANSFORMER_CONFIG: Dict[str, Any] = {
    # 序列窗口
    "window_size": 128,
    # Transformer 结构
    "d_model": 64,           # 模型维度（须能被 nhead 整除）
    "nhead": 4,              # 注意力头数
    "num_layers": 2,         # Encoder 层数
    "dim_feedforward": 128,  # FFN 隐层维度
    "dropout_rate": 0.3,     # Encoder 内部 + 全连接头 dropout（强正则化）
    # 训练超参（小数据防过拟合: 小 lr + weight_decay + 早停）
    "batch_size": 64,
    "epochs": 30,
    "learning_rate": 1e-3,
    "l2_reg": 1e-4,                 # Adam weight_decay
    "early_stop_patience": 7,
    "lr_scheduler_factor": 0.5,
    "lr_scheduler_patience": 3,
    "val_frequency": 1,             # 每个 epoch 都验证（30 epoch 内看清趋势）
    "random_state": 42,
}


# ================= 数据集 =================
class _TransformerDataset(Dataset):
    """Transformer 窗口序列数据集

    Attributes:
        X: 特征数组，形状 (n_samples, window_size, feat_dim)
        y: 标签数组，形状 (n_samples, 49)
    """

    def __init__(self, X: np.ndarray, y: np.ndarray):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.float32)

    def __len__(self) -> int:
        return len(self.X)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.X[idx], self.y[idx]


# ================= 网络结构 =================
class _TransformerNet(nn.Module):
    """TransformerEncoder 双色球全量预测网络

    结构:
        输入投影 (feat_dim -> d_model) → 可学习位置编码 → TransformerEncoder
        → 末时间步特征 → 共享 MLP → 红球头(33) + 蓝球头(16) → sigmoid cat

    Args:
        input_size: 单时间步特征维度（302）
        output_size: 输出维度（红33 + 蓝16 = 49）
        config: 模型配置字典
    """

    def __init__(self, input_size: int, output_size: int, config: Dict[str, Any]):
        super().__init__()
        self.d_model: int = config.get("d_model", 64)
        self.dropout_rate: float = config.get("dropout_rate", 0.3)
        max_len: int = config.get("window_size", 128)

        # 输入投影: 302 维特征 -> d_model 维
        self.input_proj = nn.Sequential(
            nn.Linear(input_size, self.d_model),
            nn.ReLU(),
            nn.Dropout(self.dropout_rate),
        )
        # 可学习位置编码 (1, max_len, d_model)
        self.pos_embedding = nn.Parameter(
            torch.randn(1, max_len, self.d_model) * 0.02
        )
        # Transformer Encoder（内部自带 layer_norm + dropout）
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=self.d_model,
            nhead=config.get("nhead", 4),
            dim_feedforward=config.get("dim_feedforward", 128),
            dropout=self.dropout_rate,
            batch_first=True,
            activation="relu",
        )
        self.encoder = nn.TransformerEncoder(
            encoder_layer, num_layers=config.get("num_layers", 2)
        )
        # 输出头: 共享 MLP + 双分支（红球/蓝球），仿 _HybridLSTM 的 cat 输出
        self.head = nn.Sequential(
            nn.Linear(self.d_model, 32),
            nn.ReLU(),
            nn.Dropout(self.dropout_rate),
        )
        self.red_fc = nn.Linear(32, 33)
        self.blue_fc = nn.Linear(32, 16)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """前向传播

        Args:
            x: 输入序列，形状 (batch, seq_len, feat_dim)

        Returns:
            概率输出，形状 (batch, 49)，前33为红球、后16为蓝球
        """
        if x.size(1) > self.pos_embedding.size(1):
            raise ValueError(
                f"输入序列长度 {x.size(1)} 超过位置编码最大长度 "
                f"{self.pos_embedding.size(1)}"
            )
        # (batch, seq, feat) -> (batch, seq, d_model) + 位置编码
        h = self.input_proj(x) + self.pos_embedding[:, : x.size(1), :]
        h = self.encoder(h)
        # 取末时间步作为序列表示（仿 lstm 取 h_n[-1]）
        out = self.head(h[:, -1, :])
        return torch.cat(
            [self.sigmoid(self.red_fc(out)), self.sigmoid(self.blue_fc(out))], dim=1
        )


# ================= 训练循环 =================
def _train_loop(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    criterion: nn.Module,
    optimizer: optim.Optimizer,
    epochs: int,
    device: torch.device,
    config: Dict[str, Any],
) -> Dict[str, Any]:
    """通用训练循环（ReduceLROnPlateau + 早停）

    Args:
        model: 待训练网络
        train_loader: 训练数据加载器
        val_loader: 验证数据加载器
        criterion: 损失函数（BCELoss）
        optimizer: 优化器（Adam + weight_decay）
        epochs: 最大训练轮数
        device: 计算设备
        config: 训练配置

    Returns:
        训练摘要字典: best_val_loss / epochs_trained / history
    """
    early_stop: int = config.get("early_stop_patience", 7)
    lr_factor: float = config.get("lr_scheduler_factor", 0.5)
    lr_patience: int = config.get("lr_scheduler_patience", 3)
    val_freq: int = config.get("val_frequency", 1)

    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, "min", factor=lr_factor, patience=lr_patience
    )
    best_val_loss, patience_counter = float("inf"), 0
    history: List[Dict[str, float]] = []
    t_start = time.time()
    epoch: int = -1  # 循环未执行时的保护值

    for epoch in range(epochs):
        model.train()
        total_loss = 0.0
        for inputs, targets in train_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            optimizer.zero_grad()
            loss = criterion(model(inputs), targets)
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * inputs.size(0)
        train_loss = total_loss / len(train_loader.dataset)

        if (epoch + 1) % val_freq == 0:
            model.eval()
            val_loss = 0.0
            with torch.no_grad():
                for inputs, targets in val_loader:
                    inputs, targets = inputs.to(device), targets.to(device)
                    val_loss += criterion(model(inputs), targets).item() * inputs.size(0)
            val_loss /= len(val_loader.dataset)
            scheduler.step(val_loss)
            history.append(
                {"epoch": epoch + 1, "train_loss": train_loss, "val_loss": val_loss}
            )
            if val_loss < best_val_loss:
                best_val_loss, patience_counter = val_loss, 0
            else:
                patience_counter += 1
                if patience_counter >= early_stop:
                    print(
                        f"    [early-stop] epoch {epoch + 1}, "
                        f"best_val_loss={best_val_loss:.4f}"
                    )
                    break
            print(
                f"    [epoch {epoch + 1}/{epochs}] "
                f"train_loss={train_loss:.4f} val_loss={val_loss:.4f} "
                f"lr={optimizer.param_groups[0]['lr']:.2e} "
                f"({time.time() - t_start:.0f}s)"
            )
        else:
            print(
                f"    [epoch {epoch + 1}/{epochs}] "
                f"train_loss={train_loss:.4f} ({time.time() - t_start:.0f}s)"
            )

    return {"best_val_loss": best_val_loss, "epochs_trained": epoch + 1, "history": history}


# ================= 模型实现 =================
class TransformerAllModel(BaseModel):
    """Transformer 全量预测模型（红球33 + 蓝球16）

    继承 BaseModel，仿 LSTMAllModel 的接口:
      - prepare_data: 构造 (n, window_size, feat_dim) 窗口序列 + 49 维多标签目标
      - train: Adam(weight_decay) + ReduceLROnPlateau + 早停
      - predict / predict_proba: 49 维 sigmoid 概率
      - save / load: state_dict + config.json + feature_cols.joblib + scaler.joblib

    Attributes:
        scaler: 特征标准化器（StandardScaler）
        feature_cols: 训练使用的特征列
        model: _TransformerNet 网络
    """

    def __init__(self, model_name: str = "transformer_all", config: Optional[Dict[str, Any]] = None):
        """初始化模型

        Args:
            model_name: 模型名称标识（默认 transformer_all）
            config: 模型配置字典，为 None 时使用 TRANSFORMER_CONFIG
        """
        super().__init__(model_name=model_name, config=config or TRANSFORMER_CONFIG)
        self.output_dim: int = 49  # 红球 33 + 蓝球 16
        self.model_filename: str = "transformer_all.pt"
        self.scaler: Optional[StandardScaler] = None
        self.feature_cols: Optional[List[str]] = None
        self.model: Optional[nn.Module] = None

    # ---------- 数据准备 ----------
    def prepare_data(
        self,
        df: pd.DataFrame,
        feature_cols: List[str],
        red_cols: Optional[List[str]] = None,
        blue_col: str = "Blue1",
        window_size: Optional[int] = None,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """构造窗口序列数据集（仿 LSTMAllModel.prepare_data）

        特征: 滑动窗口 (n - ws, ws, feat_dim)，用 StandardScaler 标准化；
        目标: 每行 Red1..Red6 one-hot 累加成 33 维 + Blue1 one-hot 成 16 维。

        Args:
            df: 含特征的 DataFrame（compute_all_features 产出）
            feature_cols: 特征列名列表
            red_cols: 红球列名列表，默认 RED_COLS
            blue_col: 蓝球列名
            window_size: 窗口大小，默认取配置值 128

        Returns:
            (X, y): X 形状 (n-ws, ws, feat_dim)，y 形状 (n-ws, 49)
        """
        ws: int = window_size or self.config.get("window_size", 128)
        red_cols = red_cols or RED_COLS
        self.feature_cols = feature_cols

        X_data = df[feature_cols].values.astype(np.float32)
        n = len(df)
        X = np.zeros((n - ws, ws, len(feature_cols)), dtype=np.float32)
        y = np.zeros((n - ws, 49), dtype=np.float32)
        for i in range(n - ws):
            X[i] = X_data[i : i + ws]
            red_nums = df[red_cols].iloc[i + ws].values.astype(np.int32)
            y[i, red_nums - 1] = 1.0  # 红球 one-hot 累加 -> 33 维
            y[i, 33 + int(df[blue_col].iloc[i + ws]) - 1] = 1.0  # 蓝球 -> 16 维

        # 特征标准化（仿 lstm: 展平后 fit，再还原为 3D）
        self.scaler = StandardScaler()
        return (
            self.scaler.fit_transform(X.reshape(-1, X.shape[-1])).reshape(X.shape),
            y,
        )

    # ---------- 训练 ----------
    def train(
        self,
        X_train: Union[np.ndarray, pd.DataFrame],
        y_train: Union[np.ndarray, pd.Series],
        X_val: Optional[Union[np.ndarray, pd.DataFrame]] = None,
        y_val: Optional[Union[np.ndarray, pd.Series]] = None,
    ) -> "TransformerAllModel":
        """训练模型

        Args:
            X_train: 训练窗口序列，形状 (n, ws, feat_dim)
            y_train: 训练标签，形状 (n, 49)
            X_val: 验证窗口序列（可选）
            y_val: 验证标签（可选）

        Returns:
            self: 返回自身以支持链式调用
        """
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.device = device
        # CPU 小张量性能关键: 8 线程调度开销吞掉计算收益(实测 22x 差距),
        # 4 线程为本机最优(见分析记录)。仅本模型生效, 不污染 LSTM/CNN。
        if device.type == "cpu":
            torch.set_num_threads(4)
        bs: int = self.config.get("batch_size", 64)
        epochs: int = self.config.get("epochs", 30)
        lr: float = self.config.get("learning_rate", 1e-3)

        # 构建网络（input_size 取特征维度）
        self.model = _TransformerNet(X_train.shape[-1], self.output_dim, self.config).to(device)

        train_loader = DataLoader(_TransformerDataset(X_train, y_train), batch_size=bs, shuffle=True)
        if X_val is not None and y_val is not None:
            val_loader = DataLoader(_TransformerDataset(X_val, y_val), batch_size=bs)
        else:
            # 无验证集时退化为在训练集上评估（仅用于兼容，不推荐）
            val_loader = train_loader

        optimizer = optim.Adam(
            self.model.parameters(), lr=lr, weight_decay=self.config.get("l2_reg", 1e-4)
        )
        res = _train_loop(
            self.model, train_loader, val_loader, nn.BCELoss(), optimizer,
            epochs, device, self.config,
        )

        self.metrics = {**res, "input_size": X_train.shape[-1]}
        self.is_trained = True
        return self

    # ---------- 推理 ----------
    def predict_proba(self, X: Union[np.ndarray, pd.DataFrame]) -> np.ndarray:
        """输出 49 维 sigmoid 概率（前33红球 + 后16蓝球）

        Args:
            X: 窗口序列，形状 (batch, ws, feat_dim)

        Returns:
            概率数组，形状 (batch, 49)
        """
        if not self.is_trained:
            raise RuntimeError("模型尚未训练")
        self.model.eval()
        X_t = (
            torch.tensor(X, dtype=torch.float32).to(self.device)
            if isinstance(X, np.ndarray)
            else X.to(self.device)
        )
        with torch.no_grad():
            return self.model(X_t).cpu().numpy()

    def predict_proba_normalized(self, X: Union[np.ndarray, pd.DataFrame]) -> Tuple[np.ndarray, np.ndarray]:
        """输出归一化概率（红球 softmax/33 和蓝球 softmax/16，各部分和为 1）

        Args:
            X: 窗口序列，形状 (batch, ws, feat_dim)

        Returns:
            (red_proba, blue_proba): 形状分别为 (batch, 33)、(batch, 16)
        """
        proba = self.predict_proba(X)
        red = proba[..., :33]
        blue = proba[..., 33:]
        red = red / (red.sum(axis=-1, keepdims=True) + 1e-9)
        blue = blue / (blue.sum(axis=-1, keepdims=True) + 1e-9)
        return red, blue

    def predict(self, X: Union[np.ndarray, pd.DataFrame]) -> np.ndarray:
        """按概率最大输出类别索引（兼容基类接口）

        Args:
            X: 窗口序列，形状 (batch, ws, feat_dim)

        Returns:
            类别索引数组
        """
        return np.argmax(self.predict_proba(X), axis=1)

    def predict_split(self, X: Union[np.ndarray, pd.DataFrame]) -> Tuple[np.ndarray, np.ndarray]:
        """拆分红球/蓝球概率（仿 LSTMAllModel.predict_split）

        Args:
            X: 窗口序列

        Returns:
            (red_proba, blue_proba): 形状分别为 (batch, 33)、(batch, 16)
        """
        proba = self.predict_proba(X)
        return proba[..., :33], proba[..., 33:]

    # ---------- 持久化 ----------
    def save(self, path: Optional[Union[str, Path]] = None) -> Path:
        """保存模型（state_dict + config + scaler + feature_cols + metrics）

        Args:
            path: 保存目录，为 None 时使用默认目录（saved_models/transformer_all）

        Returns:
            保存的目录路径
        """
        import joblib

        save_dir = Path(path) if path else self._get_save_dir()
        save_dir.mkdir(parents=True, exist_ok=True)
        if self.model:
            torch.save(self.model.state_dict(), save_dir / self.model_filename)
        if self.scaler:
            joblib.dump(self.scaler, save_dir / "scaler.joblib")
        if self.feature_cols:
            joblib.dump(self.feature_cols, save_dir / "feature_cols.joblib")
        with open(save_dir / "config.json", "w") as f:
            json.dump(self.config, f, default=str)
        with open(save_dir / "metrics.json", "w") as f:
            json.dump(self.metrics, f, default=str)
        return save_dir

    def load(self, path: Optional[Union[str, Path]] = None) -> "TransformerAllModel":
        """加载模型

        Args:
            path: 加载目录，为 None 时使用默认目录

        Returns:
            self: 返回自身以支持链式调用
        """
        import joblib

        load_dir = Path(path) if path else self._get_save_dir()
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.device = device

        if (load_dir / "config.json").exists():
            with open(load_dir / "config.json") as f:
                self.config = json.load(f)

        # 由 scaler 推断输入维度（与 lstm 一致）
        input_size = self.config.get("input_size", 0)
        if input_size == 0 and (load_dir / "scaler.joblib").exists():
            input_size = joblib.load(load_dir / "scaler.joblib").scale_.shape[0]

        self.model = _TransformerNet(input_size, self.output_dim, self.config).to(device)
        model_path = load_dir / self.model_filename
        if model_path.exists():
            self.model.load_state_dict(torch.load(model_path, map_location=device))
        else:
            raise FileNotFoundError(f"模型文件不存在: {model_path}")

        if (load_dir / "scaler.joblib").exists():
            self.scaler = joblib.load(load_dir / "scaler.joblib")
        if (load_dir / "feature_cols.joblib").exists():
            self.feature_cols = joblib.load(load_dir / "feature_cols.joblib")
        if (load_dir / "metrics.json").exists():
            with open(load_dir / "metrics.json") as f:
                self.metrics = json.load(f)
        self.is_trained = True
        return self
