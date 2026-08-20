"""双色球预测系统 - CNN 数学增强模型 """
import json, time, warnings
from typing import Any, Dict, List, Optional, Tuple, Union
from pathlib import Path
import numpy as np, pandas as pd
import torch, torch.nn as nn, torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from ml.config import CNN_CONFIG, RED_COLS, BLUE_COLS
from ml.models.base_model import BaseModel
warnings.filterwarnings("ignore")

class _HybridDataset(Dataset):
    def __init__(self, X, yc, yr):
        self.X, self.yc, self.yr = torch.tensor(X, dtype=torch.float32), torch.tensor(yc, dtype=torch.long), torch.tensor(yr, dtype=torch.float32)
    def __len__(self): return len(self.X)
    def __getitem__(self, i): return self.X[i].permute(1, 0), self.yc[i], self.yr[i]

class _HybridMathCNN(nn.Module):
    def __init__(self, in_ch, cfg):
        super().__init__()
        self.conv = nn.Sequential(nn.Conv1d(in_ch, cfg.get("conv_out_channels", 5), cfg.get("kernel_size", 9), padding=cfg.get("kernel_size", 9)//2),
                                  nn.BatchNorm1d(cfg.get("conv_out_channels", 5)), nn.ReLU(), nn.MaxPool1d(cfg.get("pool_size", 1)), nn.Dropout(cfg.get("dropout_rate", 0.3)))
        self.fc_h, self.rs, self.bs = cfg.get("fc_hidden_size", 5), 28, 16
        self.fc1 = self.cls_h = self.reg_h = None

    def forward(self, x):
        x = self.conv(x).view(x.size(0), -1)
        if self.fc1 is None or self.fc1.in_features != x.size(1):
            self.fc1 = nn.Linear(x.size(1), self.fc_h).to(x.device)
            self.cls_h = nn.Linear(self.fc_h, 6*self.rs + self.bs).to(x.device)
            self.reg_h = nn.Linear(self.fc_h, 7).to(x.device)
        f = torch.dropout(torch.relu(self.fc1(x)), 0.3, self.training)
        return self.cls_h(f), self.reg_h(f)

def _joint_loss(co, ro, lc, lr, w=0.01):
    ce, mse = nn.CrossEntropyLoss(), nn.MSELoss()
    cl = sum(ce(co[:, i*28:(i+1)*28], lc[:, i]) for i in range(6)) + ce(co[:, 168:184], lc[:, 6])
    rl = mse(ro, lr)
    return cl + w * rl, cl, rl

class CNNMathModel(BaseModel):
    def __init__(self, model_name: str = "cnn_reg", config=None):
        super().__init__(model_name, config or CNN_CONFIG["cnn_math"])
        self.rs, self.bs = 28, 16
        self.reg_tgts = ["Next_Sum","Next_OddRatio","Next_BigRatio","Next_Hot","Next_Cold","Next_Max_Omission","Next_Avg_Omission"]
        self._nc = self.scaler = None

    def prepare_data(self, fm, yc, yr):
        ws, step = self.config.get("window_size", 33), self.config.get("window_step", 1)
        nw = (fm.shape[0] - ws) // step + 1
        if nw <= 0: raise ValueError("数据不足")
        self._nc = fm.shape[1]
        return np.array([fm[i:i+ws-1] for i in range(nw)]), yc[ws-1::step][:nw], yr[ws-1::step][:nw]

    def train(self, X, y, Xv=None, yv=None):
        dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.device = dev
        yc, yr = y if isinstance(y, tuple) else (y, np.zeros((len(y), 7), np.float32))
        cfg = self.config
        Xt, Xv, yct, ycv, yrt, yrv = train_test_split(X, yc, yr, test_size=0.2, random_state=42)
        Xv, _, ycv, _, yrv, _ = train_test_split(Xv, ycv, yrv, test_size=0.5, random_state=42)
        tr_dl = DataLoader(_HybridDataset(Xt, yct, yrt), cfg.get("batch_size", 240), True)
        va_dl = DataLoader(_HybridDataset(Xv, ycv, yrv), cfg.get("batch_size", 240))
        
        self.model = _HybridMathCNN(self._nc or X.shape[2], cfg).to(dev)
        opt = optim.Adam(self.model.parameters(), cfg.get("learning_rate", 0.001), weight_decay=cfg.get("weight_decay", 1e-4))
        sch = optim.lr_scheduler.ReduceLROnPlateau(opt, "min", cfg.get("lr_scheduler_factor", 0.9), cfg.get("lr_scheduler_patience", 2))
        bvl, pc, ep = float("inf"), 0, cfg.get("epochs", 330)
        w, esp, li = cfg.get("regression_loss_weight", 0.01), cfg.get("early_stop_patience", 5), cfg.get("train_log_interval", 20)

        for e in range(ep):
            self.model.train()
            tl = 0.0
            for xb, ycb, yrb in tr_dl:
                xb, ycb, yrb = xb.to(dev), ycb.to(dev), yrb.to(dev)
                opt.zero_grad()
                l = _joint_loss(*self.model(xb), ycb, yrb, w)[0]
                l.backward(); opt.step()
                tl += l.item() * xb.size(0)
            tl /= len(tr_dl.dataset)
            if (e+1) % li == 0:
                self.model.eval()
                vl = 0.0
                for xb, ycb, yrb in va_dl:
                    xb, ycb, yrb = xb.to(dev), ycb.to(dev), yrb.to(dev)
                    vl += _joint_loss(*self.model(xb), ycb, yrb, w)[0].item() * xb.size(0)
                vl /= len(va_dl.dataset)
                sch.step(vl)
                if vl < bvl: bvl, pc = vl, 0
                else:
                    pc += 1
                    if pc >= esp: break
        self.metrics = {"best_val_loss": bvl, "epochs_trained": e + 1, "num_channels": self._nc}
        self.is_trained = True
        return self

    def _to_t(self, X):
        t = torch.tensor(X, dtype=torch.float32) if isinstance(X, np.ndarray) else X
        return t.unsqueeze(0).permute(0, 2, 1).to(self.device) if t.ndim == 2 else t.permute(0, 2, 1).to(self.device) if t.ndim == 3 else t.to(self.device)

    def predict_proba(self, X):
        self.model.eval()
        with torch.no_grad(): co = self.model(self._to_t(X))[0].cpu().numpy()[0]
        rp = [torch.softmax(torch.tensor(co[i*28:(i+1)*28]), 0).numpy() for i in range(6)]
        return np.array(rp), torch.softmax(torch.tensor(co[168:184]), 0).numpy()

    def predict(self, X):
        rp, bp = self.predict_proba(X)
        return np.array([np.argmax(r) + i + 1 for i, r in enumerate(rp)] + [np.argmax(bp) + 1])

    def predict_with_post_processing(self, X, df, train_stats=None):
        """带后处理的预测。

        Args:
            X: 特征矩阵
            df: 数据(含 Norm_Mean/Norm_Std/Poisson_* 等列)
            train_stats: 可选。dict 形如 {"norm_mean","norm_std","poisson_r":[33],"poisson_b":[16]}，
                提供则用**训练期**统计(避免误用未来期)。默认 None 时回退用 df.iloc[-1]
                (即预测时已知的最近一期, 非未来泄漏, 向后兼容 main.py)。
        """
        self.model.eval()
        with torch.no_grad(): co, ro = self.model(self._to_t(X))
        co, ro = co.cpu().numpy()[0], ro.cpu().numpy()[0]
        is_chaos = df.iloc[-1].get("Entropy", 0) > self.config.get("entropy_chaos_threshold", 4.0)
        damp = self.config.get("chaos_damping_factor", 0.5)
        
        reds, blue = [], 0
        for i in range(6):
            lg = co[i*28:(i+1)*28]
            if is_chaos: lg = lg * damp + np.mean(lg) * (1 - damp)
            reds.append(np.argmax(lg) + i + 1)
        lg = co[168:184]
        if is_chaos: lg = lg * damp + np.mean(lg) * (1 - damp)
        blue = np.argmax(lg) + 1

        # Norm 约束: 优先用训练期统计(防止误用未来期)
        if train_stats is not None:
            m, s = train_stats.get("norm_mean", np.nan), train_stats.get("norm_std", np.nan)
            pv = np.array(train_stats.get("poisson_r", [np.nan]*33))
        else:
            m = df.iloc[-1].get("Norm_Mean", np.nan) if "Norm_Mean" in df.columns else np.nan
            s = df.iloc[-1].get("Norm_Std", np.nan) if "Norm_Std" in df.columns else np.nan
            pv = df.iloc[-1][[f"Poisson_R{i}" for i in range(1, 34)]].values if all(f"Poisson_R{i}" in df.columns for i in range(1, 34)) else np.array([np.nan]*33)

        if not np.isnan(m):
            ps = sum(reds)
            if ps > m + 2.58*s:
                for i in range(max(reds)-1, 0, -1):
                    if i not in reds: reds[reds.index(max(reds))] = i; break
            elif ps < m - 2.58*s:
                for i in range(min(reds)+1, 34):
                    if i not in reds: reds[reds.index(min(reds))] = i; break
            reds.sort()

        pr = [f"Poisson_R{i}" for i in range(1, 34)]
        if not np.all(np.isnan(pv)):
            cands = [r if pv[r-1] > 0 else -1 for r in reds]
            for i in np.argsort(pv)[::-1]:
                if len(cands) >= 6: break
                if i+1 not in cands:
                    if -1 in cands: cands[cands.index(-1)] = i+1
                    else: cands.append(i+1)
            reds = sorted(cands[:6])
            pb = train_stats.get("poisson_b", None) if train_stats is not None else (df.iloc[-1][[f"Poisson_B{i}" for i in range(1, 17)]].values if all(f"Poisson_B{i}" in df.columns for i in range(1, 17)) else None)
            if pb is not None and pb[blue-1] == 0:
                blue = np.argmax(pb) + 1

        th = self.config.get("sum_constraint_threshold", 10)
        diff = ro[0] - sum(reds)
        if abs(diff) > th:
            if diff > 0 and max(reds) < 33:
                for i in range(max(reds)+1, 34):
                    if i not in reds: reds[reds.index(max(reds))] = i; break
            elif diff < 0 and min(reds) > 1:
                for i in range(min(reds)-1, 0, -1):
                    if i not in reds: reds[reds.index(min(reds))] = i; break
            reds.sort()

        return np.array(reds + [blue]), {n: round(float(v), 4) for n, v in zip(self.reg_tgts, ro)}

    def evaluate(self, X, y, k=6):
        self.model.eval()
        with torch.no_grad(): co = self.model(self._to_t(X))[0].cpu().numpy()[0]
        ra = [1.0 if y[i]-(i+1) in np.argsort(torch.softmax(torch.tensor(co[i*28:(i+1)*28]), 0).numpy())[-k:] else 0.0 for i in range(6)]
        ba = 1.0 if y[6]-1 in np.argsort(torch.softmax(torch.tensor(co[168:184]), 0).numpy())[-k:] else 0.0
        self.metrics = {"red_avg_top_k": np.mean(ra), "blue_top_k": ba, "overall": np.mean(ra+[ba])}
        return self.metrics

    def save(self, path=None):
        d = Path(path) if path else self._get_save_dir()
        d.mkdir(parents=True, exist_ok=True)
        if self.model: torch.save(self.model.state_dict(), d/"cnn_math.pt")
        cfg = {**self.config, "num_channels": self._nc} if self._nc else self.config
        for n, o in [("config.json", cfg), ("metrics.json", self.metrics)]:
            with open(d/n, "w") as f: json.dump(o, f, default=str)
        return d

    def load(self, path=None):
        d = Path(path) if path else self._get_save_dir()
        dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.device = dev
        if (d/"config.json").exists():
            with open(d/"config.json") as f: self.config = json.load(f)
        nc = self.config.get("num_channels", 106)
        self._nc = nc
        self.model = _HybridMathCNN(nc, self.config).to(dev)
        self.model(torch.zeros(1, nc, self.config.get("window_size", 33)-1).to(dev))
        if (d/"cnn_math.pt").exists(): self.model.load_state_dict(torch.load(d/"cnn_math.pt", map_location=dev))
        else: raise FileNotFoundError(f"模型不存在: {d/'cnn_math.pt'}")
        if (d/"metrics.json").exists():
            with open(d/"metrics.json") as f: self.metrics = json.load(f)
        self.is_trained = True
        return d
