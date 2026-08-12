"""双色球预测系统 - 红球集合预测模型（修复缺陷1）

将红球从"6个有序位置单独分类"改为"无序 6 元集合预测"：
- 输入：统一特征窗口（build_unified_features 输出）
- 网络：LSTM 编码 → 33 维 sigmoid（每个号码独立概率）
- 标签：每期红球 6 个号码 one-hot（33 维多标签）
- 损失：BCEWithLogitsLoss
- 预测：33 维概率取 top-6 作为预测集合

评估用集合命中数（预测6号 ∩ 真实6号），与随机基线(期望重叠≈1.09)对比。
"""
import json
import time
import warnings
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler

from ml.config import RED_COLS, LSTM_CONFIG
from ml.features.feature_engineer import FeatureEngineer
from ml.models.base_model import BaseModel

warnings.filterwarnings("ignore")


class _SetRedDataset(Dataset):
    def __init__(self, X: np.ndarray, y: np.ndarray):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.float32)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


class _SetRedNet(nn.Module):
    def __init__(self, input_size: int, hidden_size: int, num_layers: int, dropout: float):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size, hidden_size, num_layers,
            batch_first=True, dropout=dropout if num_layers > 1 else 0,
        )
        self.fc = nn.Linear(hidden_size, 33)

    def forward(self, x):
        _, (h_n, _) = self.lstm(x)
        return self.fc(h_n[-1])  # 33 维 logits（未 sigmoid，配合 BCEWithLogits）


class SetRedModel(BaseModel):
    """红球集合预测模型（无序 6 元集合）。"""

    def __init__(self, model_name: str = "set_red", config: Optional[dict] = None):
        cfg = {k.replace("red_", ""): v for k, v in (config or LSTM_CONFIG).items()}
        super().__init__(model_name=model_name, config=cfg or LSTM_CONFIG)
        self.red_numbers = 33
        self.scaler: Optional[StandardScaler] = None
        self.feature_cols: Optional[List[str]] = None
        self.model: Optional[nn.Module] = None

    # ---------- 数据准备 ----------
    def prepare_data(self, feature_df: pd.DataFrame, label_df: pd.DataFrame,
                     feature_cols: List[str], window_size: int = 128,
                     target_cols=None) -> Tuple[np.ndarray, np.ndarray]:
        """从统一特征窗口构造序列样本, 标签从原始截断数据取(避免泄漏)。

        Args:
            feature_df: build_unified_features 输出(不含原始 Red1..6)
            label_df: 原始截断数据(含 Red1..Red6), 用于取标签
            feature_cols: 使用的特征列名
            window_size: 序列窗口
            target_cols: 标签列(默认 RED_COLS)
        """
        ws = window_size
        self.feature_cols = feature_cols
        X_data = feature_df[feature_cols].values.astype(np.float32)
        n = len(feature_df)
        X = np.zeros((n - ws, ws, len(feature_cols)), dtype=np.float32)
        y = np.zeros((n - ws, self.red_numbers), dtype=np.float32)
        for i in range(n - ws):
            X[i] = X_data[i:i + ws]
            for num in label_df[target_cols or RED_COLS].iloc[i + ws].values:
                y[i, int(num) - 1] = 1.0
        self.scaler = StandardScaler()
        Xs = self.scaler.fit_transform(X.reshape(-1, X.shape[-1])).reshape(X.shape)
        return Xs, y

    # ---------- 训练 ----------
    def train(self, X_train, y_train, X_val=None, y_val=None, epochs=None):
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.device = device
        bs = self.config.get("batch_size", 128)
        ep = epochs or self.config.get("epochs", 60)
        lr = self.config.get("learning_rate", 0.001)

        self.model = _SetRedNet(
            X_train.shape[-1],
            self.config.get("hidden_size", 64),
            self.config.get("num_layers", 2),
            self.config.get("dropout", 0.1),
        ).to(device)

        tr_loader = DataLoader(_SetRedDataset(X_train, y_train), batch_size=bs, shuffle=True)
        va_loader = DataLoader(_SetRedDataset(X_val, y_val), batch_size=bs) if X_val is not None else tr_loader

        opt = optim.Adam(self.model.parameters(), lr=lr, weight_decay=self.config.get("l2_reg", 1e-4))
        crit = nn.BCEWithLogitsLoss()
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(opt, "min", factor=0.5, patience=3)

        best_loss, patience, t0 = float("inf"), 0, time.time()
        for e in range(ep):
            self.model.train()
            tl = 0.0
            for xb, yb in tr_loader:
                xb, yb = xb.to(device), yb.to(device)
                opt.zero_grad()
                loss = crit(self.model(xb), yb)
                loss.backward(); opt.step()
                tl += loss.item() * len(xb)
            tl /= len(tr_loader.dataset)
            # val
            self.model.eval()
            vl = 0.0
            with torch.no_grad():
                for xb, yb in va_loader:
                    xb, yb = xb.to(device), yb.to(device)
                    vl += crit(self.model(xb), yb).item() * len(xb)
            vl /= len(va_loader.dataset)
            scheduler.step(vl)
            if vl < best_loss:
                best_loss, patience = vl, 0
            else:
                patience += 1
                if patience >= self.config.get("early_stop_patience", 7):
                    break

        self.metrics = {"best_val_loss": float(best_loss), "epochs": e + 1, "elapsed": time.time() - t0}
        self.is_trained = True
        return self

    # ---------- 预测 ----------
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        if not self.is_trained or self.model is None:
            raise RuntimeError("模型尚未训练")
        self.model.eval()
        Xt = torch.tensor(X, dtype=torch.float32).to(self.device)
        with torch.no_grad():
            logits = self.model(Xt).cpu().numpy()
        return 1.0 / (1.0 + np.exp(-logits))  # sigmoid → 33 维概率

    def predict_topk(self, X: np.ndarray, k: int = 6) -> Tuple[np.ndarray, np.ndarray]:
        proba = self.predict_proba(X)
        proba = np.asarray(proba).reshape(-1)  # 统一成 (33,) 一维
        idx = np.argsort(proba)[-k:][::-1]
        return idx + 1, proba[idx]

    def predict(self, X):
        nums, _ = self.predict_topk(X, 6)
        return nums

    # ---------- 持久化 ----------
    def save(self, path=None):
        d = self._get_save_dir() if path is None else __import__("pathlib").Path(path)
        d.mkdir(parents=True, exist_ok=True)
        if self.model:
            torch.save(self.model.state_dict(), d / "set_red.pt")
        import joblib
        if self.scaler:
            joblib.dump(self.scaler, d / "scaler.joblib")
        if self.feature_cols:
            joblib.dump(self.feature_cols, d / "feature_cols.joblib")
        json.dump({"config": self.config, "metrics": self.metrics}, open(d / "meta.json", "w"), default=str)
        return d

    def load(self, path=None):
        from pathlib import Path
        import joblib
        d = self._get_save_dir() if path is None else Path(path)
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.device = device
        if (d / "meta.json").exists():
            self.config = json.load(open(d / "meta.json"))["config"]
        fc = joblib.load(d / "feature_cols.joblib") if (d / "feature_cols.joblib").exists() else None
        self.feature_cols = fc
        n_feat = len(fc) if fc else 0
        self.model = _SetRedNet(
            n_feat, self.config.get("hidden_size", 64),
            self.config.get("num_layers", 2), self.config.get("dropout", 0.1),
        ).to(device)
        self.model.load_state_dict(torch.load(d / "set_red.pt", map_location=device))
        if (d / "scaler.joblib").exists():
            self.scaler = joblib.load(d / "scaler.joblib")
        if (d / "meta.json").exists():
            self.metrics = json.load(open(d / "meta.json")).get("metrics", {})
        self.is_trained = True
        return self
