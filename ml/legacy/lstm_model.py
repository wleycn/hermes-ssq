"""
双色球预测系统 - LSTM 模型实现
包含蓝球、红球、全球三种 LSTM 模型，继承 BaseModel 接口
共享训练循环（早停 + 学习率调度），支持 PyTorch 模型持久化
"""
import time
import json
import warnings
from abc import abstractmethod
from typing import Any, Dict, List, Optional, Tuple, Union
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from ml.config import LSTM_CONFIG, MODEL_CONFIG, RED_COLS, BLUE_COLS
from ml.utils.helpers import (
    create_sequential_windows,
    top_k_accuracy,
    analyze_overfitting,
)
from ml.models.base_model import BaseModel

warnings.filterwarnings("ignore")


# ================= PyTorch 数据集 =================
class _LSTMDataset(Dataset):
    """通用 LSTM 数据集"""

    def __init__(self, X: np.ndarray, y: np.ndarray):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.float32)

    def __len__(self) -> int:
        return len(self.X)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.X[idx], self.y[idx]


# ================= 共享训练循环 =================
def _train_lstm_model(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    criterion: nn.Module,
    optimizer: optim.Optimizer,
    epochs: int,
    device: torch.device,
    config: Dict[str, Any],
) -> Dict[str, float]:
    """共享 LSTM 训练循环（早停 + 学习率调度）

    Args:
        model: PyTorch 模型
        train_loader: 训练数据加载器
        val_loader: 验证数据加载器
        criterion: 损失函数
        optimizer: 优化器
        epochs: 最大训练轮数
        device: 计算设备
        config: 训练配置

    Returns:
        训练指标字典
    """
    model.train()
    total_train_start = time.time()

    early_stop_patience = config.get("early_stop_patience", 7)
    lr_scheduler_factor = config.get("lr_scheduler_factor", 0.5)
    lr_scheduler_patience = config.get("lr_scheduler_patience", 3)
    val_frequency = config.get("val_frequency", 10)

    best_val_loss = float("inf")
    patience_counter = 0
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, "min", factor=lr_scheduler_factor, patience=lr_scheduler_patience
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

        if (epoch + 1) % val_frequency == 0:
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

            current_lr = optimizer.param_groups[0]["lr"]
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
                print(f"  ⚠ 验证损失未下降, 耐心值: {patience_counter}/{early_stop_patience}")
                if patience_counter >= early_stop_patience:
                    print(f"  ✋ 早停触发, 在 Epoch {epoch+1} 停止训练")
                    break

            model.train()

    total_train_time = time.time() - total_train_start
    print(f"\n训练完成! 总训练时间: {total_train_time:.2f}秒")

    return {
        "best_val_loss": best_val_loss,
        "total_train_time": total_train_time,
        "epochs_trained": epoch + 1,
    }


# ================= 蓝球 LSTM 模型 =================
class _ProbabilityLSTMBlue(nn.Module):
    """蓝球概率预测 LSTM 网络"""

    def __init__(self, input_size: int, config: Dict[str, Any]):
        super().__init__()
        hidden_size = config.get("blue_hidden_size", LSTM_CONFIG["blue_hidden_size"])
        num_layers = config.get("blue_num_layers", LSTM_CONFIG["blue_num_layers"])
        dropout_rate = config.get("dropout_rate", LSTM_CONFIG["dropout_rate"])

        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout_rate if num_layers > 1 else 0,
        )
        self.fc = nn.Linear(hidden_size, 16)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        _, (h_n, _) = self.lstm(x)
        last_hidden = h_n[-1]
        return self.sigmoid(self.fc(last_hidden))


class LSTMBlueModel(BaseModel):
    """LSTM 蓝球概率预测模型

    基于 ssq_03_2_lstm_blue.py 重构，预测蓝球 1-16 的出现概率。

    Args:
        model_name: 模型名称标识
        config: LSTM 配置字典
    """

    def __init__(
        self,
        model_name: str = "lstm_blue",
        config: Optional[Dict[str, Any]] = None,
    ):
        super().__init__(model_name=model_name, config=config or LSTM_CONFIG)
        self.scaler: Optional[StandardScaler] = None
        self.feature_cols: Optional[List[str]] = None

    def prepare_data(
        self,
        df: pd.DataFrame,
        feature_cols: List[str],
        target_col: str = "Blue1",
        window_size: Optional[int] = None,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """准备序列模型的滑动窗口数据

        Args:
            df: 特征工程后的 DataFrame
            feature_cols: 特征列名列表
            target_col: 目标列名
            window_size: 滑动窗口大小

        Returns:
            (X, y) 其中 X 形状为 (n_samples, window_size, n_features)
        """
        window_size = window_size or self.config.get("window_size", 128)
        self.feature_cols = feature_cols

        n = len(df)
        X_data = df[feature_cols].values
        n_features = len(feature_cols)

        X = np.zeros((n - window_size, window_size, n_features), dtype=np.float32)
        y = np.zeros((n - window_size, 16), dtype=np.float32)

        for i in range(n - window_size):
            X[i] = X_data[i:i + window_size]
            blue_num = df[target_col].iloc[i + window_size]
            y[i, int(blue_num) - 1] = 1.0

        self.scaler = StandardScaler()
        X = self.scaler.fit_transform(X.reshape(-1, X.shape[-1])).reshape(X.shape)

        print(f"蓝球 LSTM 数据准备完成 - X: {X.shape}, y: {y.shape}")
        return X, y

    def train(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: Optional[np.ndarray] = None,
        y_val: Optional[np.ndarray] = None,
    ) -> "LSTMBlueModel":
        """训练 LSTM 蓝球模型

        Args:
            X_train: 训练特征 (3D)
            y_train: 训练标签
            X_val: 验证特征
            y_val: 验证标签

        Returns:
            self
        """
        t_start = time.time()

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.device = device

        batch_size = self.config.get("batch_size", 128)
        epochs = self.config.get("epochs", 256)
        lr = self.config.get("learning_rate", 0.001)

        input_size = X_train.shape[-1]
        self.model = _ProbabilityLSTMBlue(input_size, self.config).to(device)

        train_dataset = _LSTMDataset(X_train, y_train)
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

        if X_val is not None and y_val is not None:
            val_dataset = _LSTMDataset(X_val, y_val)
            val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
        else:
            val_loader = train_loader

        criterion = nn.BCELoss()
        optimizer = optim.Adam(self.model.parameters(), lr=lr)

        train_result = _train_lstm_model(
            self.model, train_loader, val_loader, criterion, optimizer, epochs, device, self.config
        )

        self.metrics = {
            **train_result,
            "model_type": "LSTMBlue",
            "input_size": input_size,
        }

        self.is_trained = True
        elapsed = time.time() - t_start
        print(f"  ✓ LSTM 蓝球模型训练完成，耗时: {elapsed:.2f}s")
        return self

    def predict(self, X: Union[np.ndarray, torch.Tensor]) -> np.ndarray:
        """预测类别标签（取概率最高的索引）

        Args:
            X: 输入特征

        Returns:
            预测标签数组
        """
        proba = self.predict_proba(X)
        return np.argmax(proba, axis=1)

    def predict_proba(self, X: Union[np.ndarray, torch.Tensor]) -> np.ndarray:
        """预测类别概率

        Args:
            X: 输入特征

        Returns:
            预测概率数组，形状 (n_samples, 16)
        """
        if not self.is_trained or self.model is None:
            raise RuntimeError("模型尚未训练")
        self.model.eval()

        if isinstance(X, np.ndarray):
            X_tensor = torch.tensor(X, dtype=torch.float32).to(self.device)
        else:
            X_tensor = X.to(self.device)

        with torch.no_grad():
            proba = self.model(X_tensor).cpu().numpy()
        return proba

    def save(self, path: Optional[Union[str, Path]] = None) -> Path:
        save_dir = Path(path) if path else self._get_save_dir()
        save_dir.mkdir(parents=True, exist_ok=True)

        if self.model is not None:
            torch.save(self.model.state_dict(), save_dir / "lstm_blue.pt")

        if self.scaler is not None:
            import joblib
            joblib.dump(self.scaler, save_dir / "scaler.joblib")

        if self.feature_cols is not None:
            import joblib
            joblib.dump(self.feature_cols, save_dir / "feature_cols.joblib")

        with open(save_dir / "config.json", "w", encoding="utf-8") as f:
            json.dump(self.config, f, ensure_ascii=False, indent=2, default=str)

        with open(save_dir / "metrics.json", "w", encoding="utf-8") as f:
            json.dump(self.metrics, f, ensure_ascii=False, indent=2, default=str)

        print(f"✓ LSTM 蓝球模型已保存至: {save_dir}")
        return save_dir

    def load(self, path: Optional[Union[str, Path]] = None) -> "LSTMBlueModel":
        load_dir = Path(path) if path else self._get_save_dir()
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.device = device

        config_path = load_dir / "config.json"
        if config_path.exists():
            with open(config_path, "r", encoding="utf-8") as f:
                self.config = json.load(f)

        input_size = self.config.get("input_size", 0)
        if input_size == 0:
            scaler_path = load_dir / "scaler.joblib"
            import joblib
            if scaler_path.exists():
                s = joblib.load(scaler_path)
                input_size = s.scale_.shape[0]

        self.model = _ProbabilityLSTMBlue(input_size, self.config).to(device)
        model_path = load_dir / "lstm_blue.pt"
        if model_path.exists():
            self.model.load_state_dict(torch.load(model_path, map_location=device))
        else:
            raise FileNotFoundError(f"模型文件不存在: {model_path}")

        scaler_path = load_dir / "scaler.joblib"
        if scaler_path.exists():
            import joblib
            self.scaler = joblib.load(scaler_path)

        cols_path = load_dir / "feature_cols.joblib"
        if cols_path.exists():
            import joblib
            self.feature_cols = joblib.load(cols_path)

        metrics_path = load_dir / "metrics.json"
        if metrics_path.exists():
            with open(metrics_path, "r", encoding="utf-8") as f:
                self.metrics = json.load(f)

        self.is_trained = True
        print(f"✓ LSTM 蓝球模型已从 {load_dir} 加载")
        return self


# ================= 红球 LSTM 模型 =================
class _ProbabilityLSTMRed(nn.Module):
    """红球概率预测 LSTM 网络"""

    def __init__(self, input_size: int, config: Dict[str, Any]):
        super().__init__()
        hidden_size = config.get("red_hidden_size", LSTM_CONFIG["red_hidden_size"])
        num_layers = config.get("red_num_layers", LSTM_CONFIG["red_num_layers"])
        dropout_rate = config.get("red_dropout", LSTM_CONFIG["red_dropout"])

        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout_rate if num_layers > 1 else 0,
        )
        self.fc = nn.Linear(hidden_size, 33)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        _, (h_n, _) = self.lstm(x)
        last_hidden = h_n[-1]
        return self.sigmoid(self.fc(last_hidden))


class LSTMRedModel(BaseModel):
    """LSTM 红球概率预测模型

    基于 ssq_03_2_lstm_reds.py 重构，预测红球 1-33 的出现概率。

    Args:
        model_name: 模型名称标识
        config: LSTM 配置字典
    """

    def __init__(
        self,
        model_name: str = "lstm_reds",
        config: Optional[Dict[str, Any]] = None,
    ):
        super().__init__(model_name=model_name, config=config or LSTM_CONFIG)
        self.scaler: Optional[StandardScaler] = None
        self.feature_cols: Optional[List[str]] = None

    def prepare_data(
        self,
        df: pd.DataFrame,
        feature_cols: List[str],
        target_cols: List[str] = None,
        window_size: Optional[int] = None,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """准备红球滑动窗口数据（多标签）

        Args:
            df: 特征工程后的 DataFrame
            feature_cols: 特征列名列表
            target_cols: 目标列名列表
            window_size: 滑动窗口大小

        Returns:
            (X, y) 其中 y 形状为 (n_samples, 33)，多标签 multi-hot 编码
        """
        window_size = window_size or self.config.get("window_size", 128)
        target_cols = target_cols or RED_COLS
        self.feature_cols = feature_cols

        n = len(df)
        X_data = df[feature_cols].values
        n_features = len(feature_cols)

        X = np.zeros((n - window_size, window_size, n_features), dtype=np.float32)
        y = np.zeros((n - window_size, 33), dtype=np.float32)

        for i in range(n - window_size):
            X[i] = X_data[i:i + window_size]
            next_draw = df[target_cols].iloc[i + window_size].values
            for num in next_draw:
                y[i, int(num) - 1] = 1.0

        self.scaler = StandardScaler()
        X = self.scaler.fit_transform(X.reshape(-1, X.shape[-1])).reshape(X.shape)

        print(f"红球 LSTM 数据准备完成 - X: {X.shape}, y: {y.shape}")
        return X, y

    def train(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: Optional[np.ndarray] = None,
        y_val: Optional[np.ndarray] = None,
    ) -> "LSTMRedModel":
        """训练 LSTM 红球模型"""
        t_start = time.time()

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.device = device

        batch_size = self.config.get("batch_size", 128)
        epochs = self.config.get("epochs", 128)
        lr = self.config.get("learning_rate", 0.001)

        input_size = X_train.shape[-1]
        self.model = _ProbabilityLSTMRed(input_size, self.config).to(device)

        train_dataset = _LSTMDataset(X_train, y_train)
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

        if X_val is not None and y_val is not None:
            val_dataset = _LSTMDataset(X_val, y_val)
            val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
        else:
            val_loader = train_loader

        criterion = nn.BCELoss()
        optimizer = optim.Adam(self.model.parameters(), lr=lr)

        train_result = _train_lstm_model(
            self.model, train_loader, val_loader, criterion, optimizer, epochs, device, self.config
        )

        self.metrics = {
            **train_result,
            "model_type": "LSTMRed",
            "input_size": input_size,
        }

        self.is_trained = True
        elapsed = time.time() - t_start
        print(f"  ✓ LSTM 红球模型训练完成，耗时: {elapsed:.2f}s")
        return self

    def predict(self, X: Union[np.ndarray, torch.Tensor]) -> np.ndarray:
        proba = self.predict_proba(X)
        return np.argmax(proba, axis=1)

    def predict_proba(self, X: Union[np.ndarray, torch.Tensor]) -> np.ndarray:
        if not self.is_trained or self.model is None:
            raise RuntimeError("模型尚未训练")
        self.model.eval()

        if isinstance(X, np.ndarray):
            X_tensor = torch.tensor(X, dtype=torch.float32).to(self.device)
        else:
            X_tensor = X.to(self.device)

        with torch.no_grad():
            proba = self.model(X_tensor).cpu().numpy()
        return proba

    def save(self, path: Optional[Union[str, Path]] = None) -> Path:
        save_dir = Path(path) if path else self._get_save_dir()
        save_dir.mkdir(parents=True, exist_ok=True)

        if self.model is not None:
            torch.save(self.model.state_dict(), save_dir / "lstm_red.pt")

        import joblib
        if self.scaler is not None:
            joblib.dump(self.scaler, save_dir / "scaler.joblib")
        if self.feature_cols is not None:
            joblib.dump(self.feature_cols, save_dir / "feature_cols.joblib")

        with open(save_dir / "config.json", "w", encoding="utf-8") as f:
            json.dump(self.config, f, ensure_ascii=False, indent=2, default=str)
        with open(save_dir / "metrics.json", "w", encoding="utf-8") as f:
            json.dump(self.metrics, f, ensure_ascii=False, indent=2, default=str)

        print(f"✓ LSTM 红球模型已保存至: {save_dir}")
        return save_dir

    def load(self, path: Optional[Union[str, Path]] = None) -> "LSTMRedModel":
        save_dir = Path(path) if path else self._get_save_dir()
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.device = device

        import joblib
        config_path = save_dir / "config.json"
        if config_path.exists():
            with open(config_path, "r", encoding="utf-8") as f:
                self.config = json.load(f)

        input_size = self.config.get("input_size", 0)
        if input_size == 0:
            scaler_path = save_dir / "scaler.joblib"
            if scaler_path.exists():
                s = joblib.load(scaler_path)
                input_size = s.scale_.shape[0]

        self.model = _ProbabilityLSTMRed(input_size, self.config).to(device)
        model_path = save_dir / "lstm_red.pt"
        if model_path.exists():
            self.model.load_state_dict(torch.load(model_path, map_location=device))
        else:
            raise FileNotFoundError(f"模型文件不存在: {model_path}")

        scaler_path = save_dir / "scaler.joblib"
        if scaler_path.exists():
            self.scaler = joblib.load(scaler_path)
        cols_path = save_dir / "feature_cols.joblib"
        if cols_path.exists():
            self.feature_cols = joblib.load(cols_path)
        metrics_path = save_dir / "metrics.json"
        if metrics_path.exists():
            with open(metrics_path, "r", encoding="utf-8") as f:
                self.metrics = json.load(f)

        self.is_trained = True
        print(f"✓ LSTM 红球模型已从 {save_dir} 加载")
        return self


# ================= 全球 LSTM 模型（红球+蓝球） =================
class _HybridLSTM(nn.Module):
    """混合 LSTM 概率预测网络（红球+蓝球联合输出）"""

    def __init__(self, input_size: int, config: Dict[str, Any]):
        super().__init__()
        hidden_size = config.get("all_hidden_size", LSTM_CONFIG["all_hidden_size"])
        num_layers = config.get("all_num_layers", LSTM_CONFIG["all_num_layers"])
        dropout_rate = config.get("all_dropout", LSTM_CONFIG["all_dropout"])
        fc_hidden = config.get("fc_hidden_size", 32)

        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout_rate if num_layers > 1 else 0,
        )
        self.fc1 = nn.Linear(hidden_size, fc_hidden)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout_rate)
        self.red_fc = nn.Linear(fc_hidden, 33)
        self.blue_fc = nn.Linear(fc_hidden, 16)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        _, (h_n, _) = self.lstm(x)
        last_hidden = h_n[-1]
        out = self.fc1(last_hidden)
        out = self.relu(out)
        out = self.dropout(out)
        red_out = self.sigmoid(self.red_fc(out))
        blue_out = self.sigmoid(self.blue_fc(out))
        return torch.cat([red_out, blue_out], dim=1)


class LSTMAllModel(BaseModel):
    """LSTM 全球（红球+蓝球）联合预测模型

    基于 ssq_03_3_lstm_all.py 重构，同时预测红球 1-33 和蓝球 1-16。

    Args:
        model_name: 模型名称标识
        config: LSTM 配置字典
    """

    def __init__(
        self,
        model_name: str = "lstm_all",
        config: Optional[Dict[str, Any]] = None,
    ):
        super().__init__(model_name=model_name, config=config or LSTM_CONFIG)
        self.scaler: Optional[StandardScaler] = None
        self.feature_cols: Optional[List[str]] = None

    def prepare_data(
        self,
        df: pd.DataFrame,
        feature_cols: List[str],
        red_cols: List[str] = None,
        blue_col: str = "Blue1",
        window_size: Optional[int] = None,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """准备全球滑动窗口数据

        Args:
            df: 特征工程后的 DataFrame
            feature_cols: 特征列名列表
            red_cols: 红球列名列表
            blue_col: 蓝球列名
            window_size: 滑动窗口大小

        Returns:
            (X, y) 其中 y 形状为 (n_samples, 49)，前33维红球，后16维蓝球
        """
        window_size = window_size or self.config.get("window_size", 330)
        red_cols = red_cols or RED_COLS
        self.feature_cols = feature_cols

        n = len(df)
        X_data = df[feature_cols].values
        n_features = len(feature_cols)

        X = np.zeros((n - window_size, window_size, n_features), dtype=np.float32)
        y = np.zeros((n - window_size, 33 + 16), dtype=np.float32)

        for i in range(n - window_size):
            X[i] = X_data[i:i + window_size]
            red_nums = df[red_cols].iloc[i + window_size].values.astype(np.int32)
            blue_num = df[blue_col].iloc[i + window_size]
            y[i, red_nums - 1] = 1.0
            y[i, 33 + int(blue_num) - 1] = 1.0

        self.scaler = StandardScaler()
        X = self.scaler.fit_transform(X.reshape(-1, X.shape[-1])).reshape(X.shape)

        print(f"全球 LSTM 数据准备完成 - X: {X.shape}, y: {y.shape}")
        return X, y

    def train(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: Optional[np.ndarray] = None,
        y_val: Optional[np.ndarray] = None,
    ) -> "LSTMAllModel":
        """训练 LSTM 全球模型"""
        t_start = time.time()

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.device = device

        batch_size = self.config.get("batch_size", 240)
        epochs = self.config.get("epochs", 256)
        lr = self.config.get("learning_rate", 0.0003)
        l2_reg = self.config.get("l2_reg", 1e-4)

        input_size = X_train.shape[-1]
        self.model = _HybridLSTM(input_size, self.config).to(device)

        train_dataset = _LSTMDataset(X_train, y_train)
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

        if X_val is not None and y_val is not None:
            val_dataset = _LSTMDataset(X_val, y_val)
            val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
        else:
            val_loader = train_loader

        criterion = nn.BCELoss()
        optimizer = optim.Adam(self.model.parameters(), lr=lr, weight_decay=l2_reg)

        train_result = _train_lstm_model(
            self.model, train_loader, val_loader, criterion, optimizer, epochs, device, self.config
        )

        self.metrics = {
            **train_result,
            "model_type": "LSTMAll",
            "input_size": input_size,
        }

        self.is_trained = True
        elapsed = time.time() - t_start
        print(f"  ✓ LSTM 全球模型训练完成，耗时: {elapsed:.2f}s")
        return self

    def predict(self, X: Union[np.ndarray, torch.Tensor]) -> np.ndarray:
        proba = self.predict_proba(X)
        return np.argmax(proba, axis=1)

    def predict_proba(self, X: Union[np.ndarray, torch.Tensor]) -> np.ndarray:
        if not self.is_trained or self.model is None:
            raise RuntimeError("模型尚未训练")
        self.model.eval()

        if isinstance(X, np.ndarray):
            X_tensor = torch.tensor(X, dtype=torch.float32).to(self.device)
        else:
            X_tensor = X.to(self.device)

        with torch.no_grad():
            proba = self.model(X_tensor).cpu().numpy()
        return proba

    def predict_split(
        self, X: Union[np.ndarray, torch.Tensor]
    ) -> Tuple[np.ndarray, np.ndarray]:
        """分离预测红球和蓝球概率

        Args:
            X: 输入特征

        Returns:
            (red_probs, blue_probs)
        """
        proba = self.predict_proba(X)
        return proba[..., :33], proba[..., 33:]

    def save(self, path: Optional[Union[str, Path]] = None) -> Path:
        save_dir = Path(path) if path else self._get_save_dir()
        save_dir.mkdir(parents=True, exist_ok=True)

        if self.model is not None:
            torch.save(self.model.state_dict(), save_dir / "lstm_all.pt")

        import joblib
        if self.scaler is not None:
            joblib.dump(self.scaler, save_dir / "scaler.joblib")
        if self.feature_cols is not None:
            joblib.dump(self.feature_cols, save_dir / "feature_cols.joblib")

        with open(save_dir / "config.json", "w", encoding="utf-8") as f:
            json.dump(self.config, f, ensure_ascii=False, indent=2, default=str)
        with open(save_dir / "metrics.json", "w", encoding="utf-8") as f:
            json.dump(self.metrics, f, ensure_ascii=False, indent=2, default=str)

        print(f"✓ LSTM 全球模型已保存至: {save_dir}")
        return save_dir

    def load(self, path: Optional[Union[str, Path]] = None) -> "LSTMAllModel":
        save_dir = Path(path) if path else self._get_save_dir()
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.device = device

        import joblib
        config_path = save_dir / "config.json"
        if config_path.exists():
            with open(config_path, "r", encoding="utf-8") as f:
                self.config = json.load(f)

        input_size = self.config.get("input_size", 0)
        if input_size == 0:
            scaler_path = save_dir / "scaler.joblib"
            if scaler_path.exists():
                s = joblib.load(scaler_path)
                input_size = s.scale_.shape[0]

        self.model = _HybridLSTM(input_size, self.config).to(device)
        model_path = save_dir / "lstm_all.pt"
        if model_path.exists():
            self.model.load_state_dict(torch.load(model_path, map_location=device))
        else:
            raise FileNotFoundError(f"模型文件不存在: {model_path}")

        scaler_path = save_dir / "scaler.joblib"
        if scaler_path.exists():
            self.scaler = joblib.load(scaler_path)
        cols_path = save_dir / "feature_cols.joblib"
        if cols_path.exists():
            self.feature_cols = joblib.load(cols_path)
        metrics_path = save_dir / "metrics.json"
        if metrics_path.exists():
            with open(metrics_path, "r", encoding="utf-8") as f:
                self.metrics = json.load(f)

        self.is_trained = True
        print(f"✓ LSTM 全球模型已从 {save_dir} 加载")
        return self