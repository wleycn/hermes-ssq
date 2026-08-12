"""双色球预测系统 - LSTM 模型精简版"""
import json, time, warnings
from typing import Any, Dict, List, Optional, Tuple, Union
from pathlib import Path
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler

# 假设这些从外部导入
from ml.config import LSTM_CONFIG, RED_COLS, BLUE_COLS
from ml.models.base_model import BaseModel
warnings.filterwarnings("ignore")

# ================= 基础组件 =================
class _LSTMDataset(Dataset):
    def __init__(self, X: np.ndarray, y: np.ndarray):
        self.X, self.y = torch.tensor(X, dtype=torch.float32), torch.tensor(y, dtype=torch.float32)
    def __len__(self): return len(self.X)
    def __getitem__(self, idx): return self.X[idx], self.y[idx]

def _train_loop(model, train_loader, val_loader, criterion, optimizer, epochs, device, config):
    model.train()
    early_stop, lr_factor, lr_patience = config.get("early_stop_patience", 7), config.get("lr_scheduler_factor", 0.5), config.get("lr_scheduler_patience", 3)
    val_freq, best_val_loss, patience_counter = config.get("val_frequency", 10), float("inf"), 0
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, "min", factor=lr_factor, patience=lr_patience)

    for epoch in range(epochs):
        total_loss = 0.0
        for inputs, targets in train_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            optimizer.zero_grad()
            loss = criterion(model(inputs), targets)
            loss.backward(); optimizer.step()
            total_loss += loss.item() * inputs.size(0)
        
        if (epoch + 1) % val_freq == 0:
            model.eval()
            val_loss = sum(criterion(model(inputs.to(device)), targets.to(device)).item() * inputs.size(0) for inputs, targets in val_loader) / len(val_loader.dataset)
            scheduler.step(val_loss)
            if val_loss < best_val_loss: best_val_loss, patience_counter = val_loss, 0
            else:
                patience_counter += 1
                if patience_counter >= early_stop: break
            model.train()
    return {"best_val_loss": best_val_loss, "epochs_trained": epoch + 1}

# ================= 网络结构 =================
class _LSTMNet(nn.Module):
    def __init__(self, input_size: int, output_size: int, config: dict):
        super().__init__()
        h_size, n_layers = config.get("hidden_size", 64), config.get("num_layers", 2)
        self.lstm = nn.LSTM(input_size, h_size, n_layers, batch_first=True, dropout=config.get("dropout", 0.2) if n_layers > 1 else 0)
        self.fc, self.act = nn.Linear(h_size, output_size), nn.Sigmoid()
    def forward(self, x):
        _, (h_n, _) = self.lstm(x)
        return self.act(self.fc(h_n[-1]))

class _HybridLSTM(nn.Module):
    def __init__(self, input_size: int, output_size: int, config: dict):
        super().__init__()
        h_size, n_layers = config.get("hidden_size", 64), config.get("num_layers", 2)
        self.lstm = nn.LSTM(input_size, h_size, n_layers, batch_first=True, dropout=config.get("dropout", 0.2) if n_layers > 1 else 0)
        self.fc1 = nn.Sequential(nn.Linear(h_size, 32), nn.ReLU(), nn.Dropout(0.2))
        self.red_fc, self.blue_fc, self.sigmoid = nn.Linear(32, 33), nn.Linear(32, 16), nn.Sigmoid()
    def forward(self, x):
        _, (h_n, _) = self.lstm(x)
        out = self.fc1(h_n[-1])
        return torch.cat([self.sigmoid(self.red_fc(out)), self.sigmoid(self.blue_fc(out))], dim=1)

# ================= 公共基类 =================
class _BaseLSTM(BaseModel):
    def __init__(self, model_name: str, config: dict, net_class: type, output_dim: int, model_filename: str):
        super().__init__(model_name=model_name, config=config)
        self.net_class, self.output_dim, self.model_filename = net_class, output_dim, model_filename
        self.scaler: Optional[StandardScaler] = None
        self.feature_cols: Optional[List[str]] = None
        self.model: Optional[nn.Module] = None

    def train(self, X_train, y_train, X_val=None, y_val=None):
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.device = device
        bs, epochs, lr = self.config.get("batch_size", 128), self.config.get("epochs", 128), self.config.get("learning_rate", 0.001)
        
        self.model = self.net_class(X_train.shape[-1], self.output_dim, self.config).to(device)
        train_loader = DataLoader(_LSTMDataset(X_train, y_train), batch_size=bs, shuffle=True)
        val_loader = DataLoader(_LSTMDataset(X_val, y_val), batch_size=bs) if X_val is not None else train_loader
        
        opt = optim.Adam(self.model.parameters(), lr=lr, weight_decay=self.config.get("l2_reg", 0))
        res = _train_loop(self.model, train_loader, val_loader, nn.BCELoss(), opt, epochs, device, self.config)
        
        self.metrics = {**res, "input_size": X_train.shape[-1]}
        self.is_trained = True
        return self

    def predict_proba(self, X):
        if not self.is_trained: raise RuntimeError("模型尚未训练")
        self.model.eval()
        X_t = torch.tensor(X, dtype=torch.float32).to(self.device) if isinstance(X, np.ndarray) else X.to(self.device)
        with torch.no_grad(): return self.model(X_t).cpu().numpy()

    def predict(self, X): return np.argmax(self.predict_proba(X), axis=1)

    def save(self, path=None):
        save_dir = Path(path) if path else self._get_save_dir()
        save_dir.mkdir(parents=True, exist_ok=True)
        if self.model: torch.save(self.model.state_dict(), save_dir / self.model_filename)
        import joblib
        if self.scaler: joblib.dump(self.scaler, save_dir / "scaler.joblib")
        if self.feature_cols: joblib.dump(self.feature_cols, save_dir / "feature_cols.joblib")
        with open(save_dir / "config.json", "w") as f: json.dump(self.config, f, default=str)
        with open(save_dir / "metrics.json", "w") as f: json.dump(self.metrics, f, default=str)
        return save_dir

    def load(self, path=None):
        load_dir = Path(path) if path else self._get_save_dir()
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.device = device
        import joblib
        if (load_dir / "config.json").exists():
            with open(load_dir / "config.json") as f: self.config = json.load(f)
        
        input_size = self.config.get("input_size", 0)
        if input_size == 0 and (load_dir / "scaler.joblib").exists():
            input_size = joblib.load(load_dir / "scaler.joblib").scale_.shape[0]
            
        self.model = self.net_class(input_size, self.output_dim, self.config).to(device)
        model_path = load_dir / self.model_filename
        if model_path.exists(): self.model.load_state_dict(torch.load(model_path, map_location=device))
        else: raise FileNotFoundError(f"模型文件不存在: {model_path}")
        
        if (load_dir / "scaler.joblib").exists(): self.scaler = joblib.load(load_dir / "scaler.joblib")
        if (load_dir / "feature_cols.joblib").exists(): self.feature_cols = joblib.load(load_dir / "feature_cols.joblib")
        if (load_dir / "metrics.json").exists():
            with open(load_dir / "metrics.json") as f: self.metrics = json.load(f)
        self.is_trained = True
        return self

# ================= 具体模型实现 =================
class LSTMBlueModel(_BaseLSTM):
    def __init__(self, model_name="lstm_blue", config=None):
        cfg = {k.replace("blue_", ""): v for k, v in (config or LSTM_CONFIG).items()}
        super().__init__(model_name, cfg, _LSTMNet, 16, "lstm_blue.pt")

    def prepare_data(self, df, feature_cols, target_col="Blue1", window_size=None):
        ws = window_size or self.config.get("window_size", 128)
        self.feature_cols = feature_cols
        X_data = df[feature_cols].values
        X = np.zeros((len(df) - ws, ws, len(feature_cols)), dtype=np.float32)
        y = np.zeros((len(df) - ws, 16), dtype=np.float32)
        for i in range(len(df) - ws):
            X[i] = X_data[i:i + ws]
            y[i, int(df[target_col].iloc[i + ws]) - 1] = 1.0
        self.scaler = StandardScaler()
        return self.scaler.fit_transform(X.reshape(-1, X.shape[-1])).reshape(X.shape), y

class LSTMRedModel(_BaseLSTM):
    def __init__(self, model_name="lstm_reds", config=None):
        cfg = {k.replace("red_", ""): v for k, v in (config or LSTM_CONFIG).items()}
        super().__init__(model_name, cfg, _LSTMNet, 33, "lstm_red.pt")

    def prepare_data(self, df, feature_cols, target_cols=None, window_size=None):
        ws, target_cols = window_size or self.config.get("window_size", 128), target_cols or RED_COLS
        self.feature_cols = feature_cols
        X_data = df[feature_cols].values
        X = np.zeros((len(df) - ws, ws, len(feature_cols)), dtype=np.float32)
        y = np.zeros((len(df) - ws, 33), dtype=np.float32)
        for i in range(len(df) - ws):
            X[i] = X_data[i:i + ws]
            for num in df[target_cols].iloc[i + ws].values: y[i, int(num) - 1] = 1.0
        self.scaler = StandardScaler()
        return self.scaler.fit_transform(X.reshape(-1, X.shape[-1])).reshape(X.shape), y

class LSTMAllModel(_BaseLSTM):
    def __init__(self, model_name="lstm_all", config=None):
        cfg = {k.replace("all_", ""): v for k, v in (config or LSTM_CONFIG).items()}
        super().__init__(model_name, cfg, _HybridLSTM, 49, "lstm_all.pt")

    def predict_split(self, X):
        proba = self.predict_proba(X)
        return proba[..., :33], proba[..., 33:]

    def prepare_data(self, df, feature_cols, red_cols=None, blue_col="Blue1", window_size=None):
        ws, red_cols = window_size or self.config.get("window_size", 330), red_cols or RED_COLS
        self.feature_cols = feature_cols
        X_data = df[feature_cols].values
        X = np.zeros((len(df) - ws, ws, len(feature_cols)), dtype=np.float32)
        y = np.zeros((len(df) - ws, 49), dtype=np.float32)
        for i in range(len(df) - ws):
            X[i] = X_data[i:i + ws]
            red_nums = df[red_cols].iloc[i + ws].values.astype(np.int32)
            y[i, red_nums - 1] = 1.0
            y[i, 33 + int(df[blue_col].iloc[i + ws]) - 1] = 1.0
        self.scaler = StandardScaler()
        return self.scaler.fit_transform(X.reshape(-1, X.shape[-1])).reshape(X.shape), y


