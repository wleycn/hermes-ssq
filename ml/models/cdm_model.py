"""双色球预测系统 - CDM (Compound-Dirichlet-Multinomial) 贝叶斯频率模型

定位: "贝叶斯视角的频率检验器", 而非"更准的预测器"。
在双色球已被实证为独立均匀随机过程 (光谱探针 FLAT / 马尔可夫 p=0.75 /
ML 池 vs 随机池无差异) 的背景下, CDM 不做任何序列建模, 而是用
Dirichlet-Multinomial 共轭结构对每个号码的历史出现频数做贝叶斯平滑,
给出其出现概率的后验均值估计, 用于检验频数分布是否偏离均匀基准。

数学形式:
    红球: 把 6*N 次红球出现视为 33 类上的分类观测,
          先验 Dir(alpha_red, ..., alpha_red), 后验均值
          p_i = (alpha_red + count_i) / (33*alpha_red + 6*N),  sum(p) = 1
    蓝球: 把 N 次蓝球出现视为 16 类上的分类观测,
          p_i = (alpha_blue + count_i) / (16*alpha_blue + N),  sum(p) = 1

注意: 这里的 p_i 是"号码 i 在所有出现记录中的占比" (归一化到 1, 便于
与均匀基准 1/33、1/16 直接对比); 若需"单期开出号码 i 的边际概率",
红球为 6*p_i, 蓝球为 1*p_i。alpha 默认取 1.0 (均匀先验, 等价于
Laplace 平滑), 可通过 config 的 alpha_red / alpha_blue / alpha 调整。
"""
import json
from pathlib import Path
from typing import Any, Dict, Optional, Tuple, Union

import numpy as np
import pandas as pd

from ml.config import BLUE_COLS, BLUE_NUMBERS, MODEL_CONFIG, RED_COLS, RED_NUMBERS
from ml.models.base_model import BaseModel


class CDMModel(BaseModel):
    """Compound-Dirichlet-Multinomial 频率检验模型

    train() 只统计频数、不依赖特征; predict_proba() 返回 Dirichlet
    后验均值概率; save/load 用 json 持久化 counts 与先验参数。

    Attributes:
        alpha_red: 红球 Dirichlet 先验参数 (默认 1.0)
        alpha_blue: 蓝球 Dirichlet 先验参数 (默认 1.0)
        red_counts: 红球 1-33 的历史出现次数 (0-based 索引)
        blue_counts: 蓝球 1-16 的历史出现次数 (0-based 索引)
        red_total: 红球总出现次数 = 6 * n_draws
        blue_total: 蓝球总出现次数 = n_draws
        n_draws: 训练期数
    """

    def __init__(
        self, model_name: str = "cdm", config: Optional[Dict[str, Any]] = None
    ):
        """初始化 CDM 模型

        Args:
            model_name: 模型名称标识 (默认 "cdm")
            config: 模型配置字典, 可含 alpha_red / alpha_blue / alpha
                    (先验参数), 为 None 时使用默认配置
        """
        super().__init__(model_name=model_name, config=config or MODEL_CONFIG)
        # 先验参数: 优先取分项, 其次取通用 alpha, 兜底 1.0
        self.alpha_red: float = float(
            self.config.get("alpha_red", self.config.get("alpha", 1.0))
        )
        self.alpha_blue: float = float(
            self.config.get("alpha_blue", self.config.get("alpha", 1.0))
        )
        # 频数统计 (0-based 索引, red_counts[i] 对应号码 i+1)
        self.red_counts: np.ndarray = np.zeros(RED_NUMBERS, dtype=int)
        self.blue_counts: np.ndarray = np.zeros(BLUE_NUMBERS, dtype=int)
        self.red_total: int = 0
        self.blue_total: int = 0
        self.n_draws: int = 0

    # ---------- 数据提取 ----------
    def _extract_balls(
        self, X: Union[np.ndarray, pd.DataFrame]
    ) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        """从输入中提取红球/蓝球 (转为 0-based 平铺数组)

        Args:
            X: DataFrame (含 Red1..Red6 / Blue1 列) 或 ndarray,
               形状 (n, 6) 仅红球 或 (n, 7) 红球+蓝球

        Returns:
            (red, blue): red 为 0-based 红球数组 (长度 6*n),
                         blue 为 0-based 蓝球数组 (长度 n) 或 None
        """
        if isinstance(X, pd.DataFrame):
            red = X[RED_COLS].to_numpy(dtype=int).ravel() - 1
            blue_col = BLUE_COLS[0]
            blue = (
                X[blue_col].to_numpy(dtype=int).ravel() - 1
                if blue_col in X.columns
                else None
            )
        else:
            arr = np.asarray(X, dtype=int)
            if arr.ndim != 2 or arr.shape[1] < 6:
                raise ValueError(f"ndarray 输入需为 (n,6) 或 (n,7), 实际形状 {arr.shape}")
            red = arr[:, :6].ravel() - 1
            blue = arr[:, 6].ravel() - 1 if arr.shape[1] >= 7 else None
        return red, blue

    # ---------- 接口实现 ----------
    def train(
        self,
        X_train: Union[np.ndarray, pd.DataFrame],
        y_train: Optional[Union[np.ndarray, pd.Series]] = None,
        X_val: Optional[Union[np.ndarray, pd.DataFrame]] = None,
        y_val: Optional[Union[np.ndarray, pd.Series]] = None,
    ) -> "CDMModel":
        """统计历史频数并计算 Dirichlet 后验参数

        Args:
            X_train: 开奖数据, DataFrame (含 Red1..Red6 / Blue1 列)
                     或 ndarray (n,6) 仅红球 / (n,7) 红球+蓝球
            y_train: 不使用 (CDM 为无监督频数模型), 仅为接口兼容
            X_val: 不使用, 仅为接口兼容
            y_val: 不使用, 仅为接口兼容

        Returns:
            self: 返回自身以支持链式调用
        """
        red, blue = self._extract_balls(X_train)
        self.n_draws = len(X_train)
        # 红球总出现次数 = 每期 6 个红球
        self.red_total = int(red.size)
        self.red_counts = np.bincount(red, minlength=RED_NUMBERS)[:RED_NUMBERS]
        # 蓝球统计 (无蓝球列时保持全 0, 后验退化为均匀先验)
        self.blue_counts = np.zeros(BLUE_NUMBERS, dtype=int)
        if blue is not None:
            self.blue_total = int(blue.size)
            self.blue_counts = np.bincount(blue, minlength=BLUE_NUMBERS)[:BLUE_NUMBERS]
        self.is_trained = True
        return self

    def predict_proba(self, X: Optional[Union[np.ndarray, pd.DataFrame]] = None) -> np.ndarray:
        """返回各号码的 Dirichlet 后验均值概率

        Args:
            X: 可选。CDM 不依赖特征, 传入仅用于对齐 (n_samples, n_classes)
               接口; 为 None 时返回形状 (1, 49)

        Returns:
            shape (n, 49) 的概率数组: [:, :33] 为红球概率 (和=1),
            [:, 33:] 为蓝球概率 (和=1)
        """
        if not self.is_trained:
            raise RuntimeError("模型尚未训练, 请先调用 train()")
        # 后验均值: p_i = (alpha + count_i) / (K*alpha + total)
        p_red = (self.alpha_red + self.red_counts) / (
            RED_NUMBERS * self.alpha_red + self.red_total
        )
        p_blue = (self.alpha_blue + self.blue_counts) / (
            BLUE_NUMBERS * self.alpha_blue + self.blue_total
        )
        proba = np.concatenate([p_red, p_blue]).reshape(1, -1)
        if X is not None:
            proba = np.repeat(proba, len(X), axis=0)
        return proba

    def predict(self, X: Optional[Union[np.ndarray, pd.DataFrame]] = None) -> np.ndarray:
        """预测类别标签 (红蓝拼接向量的 argmax, 接口兼容用)

        Args:
            X: 可选输入, 同上

        Returns:
            每行概率最大类别下标 (0-48)
        """
        return np.argmax(self.predict_proba(X), axis=1)

    # ---------- 选号辅助 ----------
    def top_numbers(
        self, k_red: int = 6, k_blue: int = 1
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """按后验概率取红球/蓝球 Top-K 号码

        Args:
            k_red: 红球取前 K 个 (默认 6)
            k_blue: 蓝球取前 K 个 (默认 1)

        Returns:
            (red_nums, red_probs, blue_nums, blue_probs):
            均为 1-based 号码数组与对应后验概率数组
        """
        proba = self.predict_proba()[0]
        p_red, p_blue = proba[:RED_NUMBERS], proba[RED_NUMBERS:]
        red_idx = np.argsort(p_red)[::-1][:k_red]
        blue_idx = np.argsort(p_blue)[::-1][:k_blue]
        return (
            red_idx + 1,
            p_red[red_idx],
            blue_idx + 1,
            p_blue[blue_idx],
        )

    # ---------- 持久化 ----------
    def save(self, path: Optional[Union[str, Path]] = None) -> Path:
        """用 json 保存频数统计与先验参数

        Args:
            path: 保存目录, 为 None 时使用默认目录 (MODELS_DIR/cdm)

        Returns:
            保存的目录路径
        """
        save_dir = Path(path) if path else self._get_save_dir()
        save_dir.mkdir(parents=True, exist_ok=True)
        payload = {
            "model_name": self.model_name,
            "alpha_red": self.alpha_red,
            "alpha_blue": self.alpha_blue,
            "red_counts": self.red_counts.tolist(),
            "blue_counts": self.blue_counts.tolist(),
            "red_total": self.red_total,
            "blue_total": self.blue_total,
            "n_draws": self.n_draws,
            "is_trained": self.is_trained,
            "metrics": self.metrics,
        }
        with open(save_dir / "cdm_model.json", "w", encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=False, indent=2, default=str)
        return save_dir

    def load(self, path: Optional[Union[str, Path]] = None) -> "CDMModel":
        """从 json 加载模型状态

        Args:
            path: 加载目录, 为 None 时使用默认目录 (MODELS_DIR/cdm)

        Returns:
            self: 返回自身以支持链式调用
        """
        load_dir = Path(path) if path else self._get_save_dir()
        model_file = load_dir / "cdm_model.json"
        if not model_file.exists():
            raise FileNotFoundError(f"CDM 模型文件不存在: {model_file}")
        with open(model_file, "r", encoding="utf-8") as f:
            payload = json.load(f)
        self.model_name = payload["model_name"]
        self.alpha_red = float(payload["alpha_red"])
        self.alpha_blue = float(payload["alpha_blue"])
        self.red_counts = np.asarray(payload["red_counts"], dtype=int)
        self.blue_counts = np.asarray(payload["blue_counts"], dtype=int)
        self.red_total = int(payload["red_total"])
        self.blue_total = int(payload["blue_total"])
        self.n_draws = int(payload["n_draws"])
        self.is_trained = bool(payload["is_trained"])
        self.metrics = payload.get("metrics", {})
        return self
