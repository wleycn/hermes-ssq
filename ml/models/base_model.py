"""
双色球预测系统 - 模型抽象基类
定义所有预测模型的通用接口和共享逻辑
"""
import time
import json
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd

from ml.config import MODEL_CONFIG, MODELS_DIR
from ml.utils.helpers import top_k_accuracy, analyze_overfitting, print_model_report


class BaseModel(ABC):
    """双色球预测模型抽象基类

    所有模型必须实现 train(), predict(), predict_proba(), save(), load() 方法。
    提供通用的评估、过拟合分析、结果保存等共享逻辑。

    Attributes:
        model_name: 模型名称标识
        config: 模型配置字典
        model: 内部模型对象（sklearn/PyTorch/LightGBM）
        label_encoder: 标签编码器
        is_trained: 模型是否已训练
        metrics: 训练评估指标
        device: 计算设备（PyTorch 模型使用）
    """

    def __init__(self, model_name: str, config: Optional[Dict[str, Any]] = None):
        """初始化基类

        Args:
            model_name: 模型名称标识
            config: 模型配置字典，为None时使用默认配置
        """
        self.model_name = model_name
        self.config = config or MODEL_CONFIG
        self.model: Any = None
        self.label_encoder: Any = None
        self.is_trained: bool = False
        self.metrics: Dict[str, Any] = {}
        self.device: Optional[Any] = None
        self._X_train_columns: Optional[List[str]] = None

    @abstractmethod
    def train(
        self,
        X_train: Union[np.ndarray, pd.DataFrame],
        y_train: Union[np.ndarray, pd.Series],
        X_val: Optional[Union[np.ndarray, pd.DataFrame]] = None,
        y_val: Optional[Union[np.ndarray, pd.Series]] = None,
    ) -> "BaseModel":
        """训练模型

        Args:
            X_train: 训练特征
            y_train: 训练标签
            X_val: 验证特征（可选）
            y_val: 验证标签（可选）

        Returns:
            self: 返回自身以支持链式调用
        """
        pass

    @abstractmethod
    def predict(self, X: Union[np.ndarray, pd.DataFrame]) -> np.ndarray:
        """预测类别标签

        Args:
            X: 输入特征

        Returns:
            预测标签数组
        """
        pass

    @abstractmethod
    def predict_proba(self, X: Union[np.ndarray, pd.DataFrame]) -> np.ndarray:
        """预测类别概率

        Args:
            X: 输入特征

        Returns:
            预测概率数组，形状 (n_samples, n_classes)
        """
        pass

    def evaluate(
        self,
        X_test: Union[np.ndarray, pd.DataFrame],
        y_test: Union[np.ndarray, pd.Series],
        k: int = 6,
    ) -> Dict[str, Any]:
        """通用评估逻辑

        计算 Top-K 命中率、对数损失、过拟合分析等指标。

        Args:
            X_test: 测试特征
            y_test: 测试标签
            k: Top-K 参数

        Returns:
            评估指标字典
        """
        if not self.is_trained:
            raise RuntimeError("模型尚未训练，请先调用 train() 方法")

        y_pred_proba = self.predict_proba(X_test)
        y_pred = self.predict(X_test)

        top_k_acc = top_k_accuracy(
            np.asarray(y_test), y_pred_proba, k=k
        )

        from sklearn.metrics import accuracy_score, log_loss

        all_classes = np.arange(y_pred_proba.shape[1])
        try:
            ll = log_loss(np.asarray(y_test), y_pred_proba, labels=all_classes)
        except Exception:
            ll = float("nan")

        acc = accuracy_score(np.asarray(y_test), y_pred)

        metrics = {
            "model_name": self.model_name,
            "test_size": len(y_test),
            "top_k_accuracy": float(top_k_acc),
            "accuracy": float(acc),
            "log_loss": float(ll),
        }

        self.metrics = metrics
        return metrics

    def save(self, path: Optional[Union[str, Path]] = None) -> Path:
        """保存模型及相关工件

        Args:
            path: 保存路径，为None时使用默认路径

        Returns:
            保存的目录路径
        """
        raise NotImplementedError("子类必须实现 save() 方法")

    def load(self, path: Optional[Union[str, Path]] = None) -> "BaseModel":
        """加载模型及相关工件

        Args:
            path: 加载路径，为None时使用默认路径

        Returns:
            self: 返回自身以支持链式调用
        """
        raise NotImplementedError("子类必须实现 load() 方法")

    def _get_save_dir(self) -> Path:
        """获取默认保存目录

        Returns:
            模型保存目录路径
        """
        save_dir = MODELS_DIR / self.model_name
        save_dir.mkdir(parents=True, exist_ok=True)
        return save_dir

    def print_report(self, metrics: Optional[Dict[str, Any]] = None) -> None:
        """打印模型评估报告

        Args:
            metrics: 评估指标字典，为None时使用 self.metrics
        """
        metrics = metrics or self.metrics
        if metrics:
            print_model_report(metrics, model_name=self.model_name)

    def __repr__(self) -> str:
        status = "已训练" if self.is_trained else "未训练"
        return f"{self.__class__.__name__}(name={self.model_name}, status={status})"