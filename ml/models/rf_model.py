"""
双色球预测系统 - 随机森林模型实现
基于 ssq_01_rf.py 的重构版本，继承 BaseModel 接口
支持滑动窗口数据管道、网格搜索超参数调优、Top-K 概率预测
"""
import time
import json
from typing import Any, Dict, List, Optional, Tuple, Union
from pathlib import Path

import numpy as np
import pandas as pd
import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, KFold
from sklearn.metrics import accuracy_score, log_loss
from sklearn.preprocessing import LabelEncoder

from ml.config import RF_CONFIG, MODEL_CONFIG, FEATURE_CONFIG
from ml.utils.helpers import (
    sliding_window_numpy,
    top_k_accuracy,
    analyze_overfitting,
)
from ml.models.base_model import BaseModel


class RandomForestModel(BaseModel):
    """随机森林预测模型

    封装 sklearn RandomForestClassifier，支持：
    - 滑动窗口数据准备与标签编码
    - 基础模型训练与评估
    - 手动网格搜索调参（避免 make_scorer 兼容性问题）
    - 模型持久化（joblib）

    Args:
        model_name: 模型名称标识
        config: 模型配置字典，为None时使用 RF_CONFIG
    """

    def __init__(
        self,
        model_name: str = "rf",
        config: Optional[Dict[str, Any]] = None,
    ):
        """初始化随机森林模型

        Args:
            model_name: 模型名称标识
            config: 模型配置字典
        """
        super().__init__(model_name=model_name, config=config or RF_CONFIG)
        self._param_grid: List[Dict[str, Any]] = self._build_param_grid()

    def _build_param_grid(self) -> List[Dict[str, Any]]:
        """从配置构建参数网格组合

        Returns:
            参数组合列表
        """
        grid = self.config
        params: List[Dict[str, Any]] = []
        for n_est in grid.get("n_estimators", [50]):
            for max_d in grid.get("max_depth", [5]):
                for min_split in grid.get("min_samples_split", [3]):
                    for min_leaf in grid.get("min_samples_leaf", [12]):
                        params.append({
                            "n_estimators": n_est,
                            "max_depth": max_d,
                            "min_samples_split": min_split,
                            "min_samples_leaf": min_leaf,
                        })
        return params

    def prepare_data(
        self,
        series: pd.Series,
        window_size: Optional[int] = None,
        step: Optional[int] = None,
        min_label_count: Optional[int] = None,
    ) -> Tuple[pd.DataFrame, np.ndarray, LabelEncoder, int]:
        """滑动窗口数据准备与标签编码

        Args:
            series: 原始数据序列
            window_size: 滑动窗口大小
            step: 滑动步长
            min_label_count: 标签最少出现次数

        Returns:
            (X, y_encoded, label_encoder, num_classes)
        """
        window_size = window_size or FEATURE_CONFIG["window_size"]
        step = step or FEATURE_CONFIG["window_step"]
        min_label_count = min_label_count or FEATURE_CONFIG["min_label_count"]

        t_start = time.time()

        r1 = sliding_window_numpy(series, window_size=window_size, step=step)
        last_col_name = r1.columns[-1]
        r1 = r1.rename(columns={last_col_name: "label"})

        X = r1.iloc[:, :-1]
        y_raw = r1.iloc[:, -1]

        value_counts = y_raw.value_counts()
        valid_labels = value_counts[value_counts >= min_label_count].index
        valid_mask = y_raw.isin(valid_labels)

        X = X[valid_mask].reset_index(drop=True)
        y_raw = y_raw[valid_mask].reset_index(drop=True)

        self.label_encoder = LabelEncoder()
        y_encoded = self.label_encoder.fit_transform(y_raw)
        num_classes = len(self.label_encoder.classes_)

        self._X_train_columns = list(X.columns)

        elapsed = time.time() - t_start

        print(f"  ✓ 数据准备完成")
        print(f"    特征维度: {X.shape}, 类别数: {num_classes}, 有效样本: {len(X)}")
        print(f"    耗时: {elapsed:.2f}s")

        return X, y_encoded, self.label_encoder, num_classes

    def train(
        self,
        X_train: Union[np.ndarray, pd.DataFrame],
        y_train: Union[np.ndarray, pd.Series],
        X_val: Optional[Union[np.ndarray, pd.DataFrame]] = None,
        y_val: Optional[Union[np.ndarray, pd.Series]] = None,
    ) -> "RandomForestModel":
        """训练随机森林模型（自动网格搜索调优）

        Args:
            X_train: 训练特征
            y_train: 训练标签
            X_val: 验证特征（忽略，使用交叉验证）
            y_val: 验证标签（忽略，使用交叉验证）

        Returns:
            self
        """
        t_start = time.time()

        X_train, X_test, y_train, y_test = train_test_split(
            X_train,
            y_train,
            test_size=MODEL_CONFIG["test_size"],
            random_state=MODEL_CONFIG["random_state"],
            stratify=y_train,
        )

        best_params, best_score, _ = self._grid_search(X_train, y_train)

        self.model = RandomForestClassifier(
            n_estimators=best_params["n_estimators"],
            max_depth=best_params["max_depth"],
            min_samples_split=best_params["min_samples_split"],
            min_samples_leaf=best_params["min_samples_leaf"],
            random_state=MODEL_CONFIG["random_state"],
            n_jobs=MODEL_CONFIG.get("n_jobs", -1),
            verbose=0,
            oob_score=True,
        )
        self.model.fit(X_train, y_train)

        y_pred_proba_train = self.model.predict_proba(X_train)
        y_pred_proba_test = self.model.predict_proba(X_test)

        all_classes = np.arange(y_pred_proba_test.shape[1])
        train_log_loss = log_loss(y_train, y_pred_proba_train, labels=all_classes)
        test_log_loss = log_loss(y_test, y_pred_proba_test, labels=all_classes)

        train_top_k_acc = top_k_accuracy(y_train, y_pred_proba_train)
        test_top_k_acc = top_k_accuracy(y_test, y_pred_proba_test)

        train_accuracy = accuracy_score(y_train, self.model.predict(X_train))
        test_accuracy = accuracy_score(y_test, self.model.predict(X_test))

        overfit_analysis = analyze_overfitting(
            train_log_loss, test_log_loss, train_top_k_acc, test_top_k_acc
        )

        elapsed = time.time() - t_start

        self.metrics = {
            "best_params": best_params,
            "cv_mean_score": best_score,
            "train_log_loss": float(train_log_loss),
            "test_log_loss": float(test_log_loss),
            "train_top_k_acc": float(train_top_k_acc),
            "test_top_k_acc": float(test_top_k_acc),
            "train_accuracy": float(train_accuracy),
            "test_accuracy": float(test_accuracy),
            "oob_score": float(self.model.oob_score_)
            if hasattr(self.model, "oob_score_")
            else None,
            "overfit_status": overfit_analysis["status"],
            "overfit_severity": overfit_analysis["severity"],
            "overfit_reason": overfit_analysis["reason"],
            "overfit_suggestion": overfit_analysis["suggestion"],
            "elapsed": elapsed,
        }

        self.is_trained = True
        print(f"  ✓ 模型训练完成，耗时: {elapsed:.2f}s")
        return self

    def _grid_search(
        self,
        X_train: Union[np.ndarray, pd.DataFrame],
        y_train: Union[np.ndarray, pd.Series],
    ) -> Tuple[Dict[str, Any], float, List[float]]:
        """手动网格搜索超参数调优

        Args:
            X_train: 训练特征
            y_train: 训练标签

        Returns:
            (best_params, best_score, all_cv_scores)
        """
        kfold = KFold(
            n_splits=3,
            shuffle=True,
            random_state=MODEL_CONFIG["random_state"],
        )

        best_score = -1.0
        best_params = self.config.get("base_params", {})
        all_cv_scores: List[float] = []

        print(f"  正在网格搜索调参 (共 {len(self._param_grid)} 组参数)...")

        for params in self._param_grid:
            cv_scores = []

            for train_idx, val_idx in kfold.split(X_train, y_train):
                X_tr = X_train.iloc[train_idx] if isinstance(X_train, pd.DataFrame) else X_train[train_idx]
                y_tr = y_train[train_idx]
                X_val = X_train.iloc[val_idx] if isinstance(X_train, pd.DataFrame) else X_train[val_idx]
                y_val = y_train[val_idx]

                model = RandomForestClassifier(
                    n_estimators=params["n_estimators"],
                    max_depth=params["max_depth"],
                    min_samples_split=params["min_samples_split"],
                    min_samples_leaf=params["min_samples_leaf"],
                    random_state=MODEL_CONFIG["random_state"],
                    n_jobs=MODEL_CONFIG.get("n_jobs", -1),
                    verbose=0,
                )
                model.fit(X_tr, y_tr)

                y_pred_proba_val = model.predict_proba(X_val)
                val_score = top_k_accuracy(y_val, y_pred_proba_val)
                cv_scores.append(val_score)

            mean_cv = np.mean(cv_scores)
            all_cv_scores.append(mean_cv)

            if mean_cv > best_score:
                best_score = mean_cv
                best_params = params.copy()

        return best_params, best_score, all_cv_scores

    def predict(self, X: Union[np.ndarray, pd.DataFrame]) -> np.ndarray:
        """预测类别标签

        Args:
            X: 输入特征

        Returns:
            预测标签数组
        """
        if not self.is_trained or self.model is None:
            raise RuntimeError("模型尚未训练")
        return self.model.predict(X)

    def predict_proba(self, X: Union[np.ndarray, pd.DataFrame]) -> np.ndarray:
        """预测类别概率

        Args:
            X: 输入特征

        Returns:
            预测概率数组，形状 (n_samples, n_classes)
        """
        if not self.is_trained or self.model is None:
            raise RuntimeError("模型尚未训练")
        return self.model.predict_proba(X)

    def predict_top_k(
        self,
        X: Union[np.ndarray, pd.DataFrame],
        k: int = 6,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """预测 Top-K 号码及其概率

        Args:
            X: 输入特征
            k: 返回前K个候选

        Returns:
            (top_numbers, top_probs)
        """
        proba = self.predict_proba(X)
        if proba.ndim == 2:
            proba = proba[0]
        top_indices = np.argsort(proba)[-k:][::-1]
        top_numbers = self.label_encoder.inverse_transform(top_indices)
        top_probs = proba[top_indices]
        return top_numbers, top_probs

    def save(self, path: Optional[Union[str, Path]] = None) -> Path:
        """保存模型及标签编码器

        Args:
            path: 保存路径，为None时使用默认路径

        Returns:
            保存的目录路径
        """
        save_dir = Path(path) if path else self._get_save_dir()
        save_dir.mkdir(parents=True, exist_ok=True)

        if self.model is not None:
            joblib.dump(self.model, save_dir / "rf_model.joblib")

        if self.label_encoder is not None:
            joblib.dump(self.label_encoder, save_dir / "label_encoder.joblib")

        if self._X_train_columns is not None:
            joblib.dump(self._X_train_columns, save_dir / "columns.joblib")

        metrics_path = save_dir / "metrics.json"
        with open(metrics_path, "w", encoding="utf-8") as f:
            json.dump(self.metrics, f, ensure_ascii=False, indent=2, default=str)

        print(f"✓ 随机森林模型已保存至: {save_dir}")
        return save_dir

    def load(self, path: Optional[Union[str, Path]] = None) -> "RandomForestModel":
        """加载模型及标签编码器

        Args:
            path: 加载路径，为None时使用默认路径

        Returns:
            self
        """
        load_dir = Path(path) if path else self._get_save_dir()

        model_path = load_dir / "rf_model.joblib"
        encoder_path = load_dir / "label_encoder.joblib"
        columns_path = load_dir / "columns.joblib"

        if model_path.exists():
            self.model = joblib.load(model_path)
        else:
            raise FileNotFoundError(f"模型文件不存在: {model_path}")

        if encoder_path.exists():
            self.label_encoder = joblib.load(encoder_path)

        if columns_path.exists():
            self._X_train_columns = joblib.load(columns_path)

        metrics_path = load_dir / "metrics.json"
        if metrics_path.exists():
            with open(metrics_path, "r", encoding="utf-8") as f:
                self.metrics = json.load(f)

        self.is_trained = True
        print(f"✓ 随机森林模型已从 {load_dir} 加载")
        return self