"""
双色球预测系统 - LightGBM 模型实现
基于 ssq_02_lgb.py 的重构版本，继承 BaseModel 接口
集成 FeatureEngineer 特征工程、支持早停和模型持久化
"""
import os
import time
import json
from typing import Any, Dict, List, Optional, Tuple, Union
from pathlib import Path

import numpy as np
import pandas as pd
import lightgbm as lgb
import joblib
from scipy.stats import skew, kurtosis
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.metrics import accuracy_score, log_loss
from sklearn.preprocessing import LabelEncoder

from ml.config import LGB_CONFIG, MODEL_CONFIG, FEATURE_CONFIG, RED_COLS
from ml.features.feature_engineer import FeatureEngineer
from ml.utils.helpers import (
    sliding_window_numpy,
    top_k_accuracy,
    analyze_overfitting,
)
from ml.models.base_model import BaseModel

PRIMES = {2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31}


def _calc_statistical_features(window_data: pd.DataFrame) -> pd.DataFrame:
    features: Dict[str, np.ndarray] = {}
    row_values = window_data.values
    features["Sum"] = row_values.sum(axis=1)
    features["Mean"] = row_values.mean(axis=1)
    features["Std"] = row_values.std(axis=1)
    features["Min"] = row_values.min(axis=1)
    features["Max"] = row_values.max(axis=1)
    features["Range"] = features["Max"] - features["Min"]
    features["Skew"] = skew(row_values, axis=1)
    features["Kurtosis"] = kurtosis(row_values, axis=1)
    return pd.DataFrame(features)


def _calc_frequency_features(window_data: pd.DataFrame) -> pd.DataFrame:
    features: Dict[str, np.ndarray] = {}
    row_values = window_data.values
    for num in range(1, 34):
        features[f"Freq_{num}"] = (row_values == num).sum(axis=1)
    features["Unique_Count"] = pd.DataFrame(row_values).nunique(axis=1).values
    return pd.DataFrame(features)


def _calc_interval_features(window_data: pd.DataFrame) -> pd.DataFrame:
    features: Dict[str, np.ndarray] = {}
    row_values = window_data.values
    features["Int_1_11"] = ((row_values >= 1) & (row_values <= 11)).sum(axis=1)
    features["Int_12_22"] = ((row_values >= 12) & (row_values <= 22)).sum(axis=1)
    features["Int_23_33"] = ((row_values >= 23) & (row_values <= 33)).sum(axis=1)
    features["Int_Max"] = np.max([features["Int_1_11"], features["Int_12_22"], features["Int_23_33"]], axis=0)
    features["Int_Min"] = np.min([features["Int_1_11"], features["Int_12_22"], features["Int_23_33"]], axis=0)
    return pd.DataFrame(features)


def _calc_odd_even_features(window_data: pd.DataFrame) -> pd.DataFrame:
    features: Dict[str, np.ndarray] = {}
    row_values = window_data.values
    features["Odd_Count"] = (row_values % 2 == 1).sum(axis=1)
    features["Even_Count"] = (row_values % 2 == 0).sum(axis=1)
    features["Odd_Even_Ratio"] = features["Odd_Count"] / (features["Even_Count"] + 1e-6)
    return pd.DataFrame(features)


def _calc_size_features(window_data: pd.DataFrame) -> pd.DataFrame:
    features: Dict[str, np.ndarray] = {}
    row_values = window_data.values
    features["Big_Count"] = (row_values > 16).sum(axis=1)
    features["Small_Count"] = (row_values <= 16).sum(axis=1)
    features["Big_Small_Ratio"] = features["Big_Count"] / (features["Small_Count"] + 1e-6)
    return pd.DataFrame(features)


def _calc_consecutive_features(window_data: pd.DataFrame) -> pd.DataFrame:
    features: Dict[str, List] = {"Consecutive_Pairs": [], "Max_Consecutive": []}
    row_values = window_data.values
    for row in row_values:
        sorted_row = np.sort(row)
        diffs = np.diff(sorted_row)
        consecutive_pairs = int(np.sum(diffs == 1))
        max_consecutive = 1
        current = 1
        for d in diffs:
            if d == 1:
                current += 1
                max_consecutive = max(max_consecutive, current)
            else:
                current = 1
        features["Consecutive_Pairs"].append(consecutive_pairs)
        features["Max_Consecutive"].append(max_consecutive)
    features["Consecutive_Ratio"] = np.array(features["Consecutive_Pairs"]) / (len(window_data.columns) - 1)
    return pd.DataFrame(features)


def _calc_prime_features(window_data: pd.DataFrame) -> pd.DataFrame:
    features: Dict[str, np.ndarray] = {}
    row_values = window_data.values
    features["Prime_Count"] = np.isin(row_values, list(PRIMES)).sum(axis=1)
    features["Prime_Ratio"] = features["Prime_Count"] / len(window_data.columns)
    return pd.DataFrame(features)


def _calc_position_features(window_data: pd.DataFrame) -> pd.DataFrame:
    features: Dict[str, np.ndarray] = {}
    row_values = window_data.values
    features["Max_Position"] = np.argmax(row_values, axis=1)
    features["Min_Position"] = np.argmin(row_values, axis=1)
    median_positions: List[int] = []
    for row in row_values:
        sorted_indices = np.argsort(row)
        median_positions.append(int(sorted_indices[len(row) // 2]))
    features["Median_Position"] = np.array(median_positions)
    return pd.DataFrame(features)


def calculate_all_features(window_data: pd.DataFrame) -> pd.DataFrame:
    """计算所有特征

    Args:
        window_data: 滑动窗口数据DataFrame

    Returns:
        拼接后的特征DataFrame
    """
    all_features = [window_data]
    feature_funcs = [
        ("统计特征", _calc_statistical_features),
        ("频率特征", _calc_frequency_features),
        ("区间特征", _calc_interval_features),
        ("奇偶特征", _calc_odd_even_features),
        ("大小特征", _calc_size_features),
        ("连号特征", _calc_consecutive_features),
        ("质数特征", _calc_prime_features),
        ("位置特征", _calc_position_features),
    ]
    for name, func in feature_funcs:
        try:
            feat_df = func(window_data)
            all_features.append(feat_df)
        except Exception as e:
            print(f"  ⚠️ {name}计算失败: {e}")
    return pd.concat(all_features, axis=1)


class LightGBMModel(BaseModel):
    """LightGBM 预测模型

    封装 LightGBM 梯度提升机，支持：
    - 滑动窗口数据准备与标签编码
    - 特征工程增强（统计/频率/区间等8类特征）
    - 基础模型训练（带早停）
    - 手动网格搜索调参
    - 模型持久化（joblib）

    Args:
        model_name: 模型名称标识
        config: 模型配置字典
    """

    def __init__(
        self,
        model_name: str = "lgbm",
        config: Optional[Dict[str, Any]] = None,
    ):
        """初始化 LightGBM 模型

        Args:
            model_name: 模型名称标识
            config: 模型配置字典
        """
        super().__init__(model_name=model_name, config=config or LGB_CONFIG)
        self._param_grid: List[Dict[str, Any]] = self._build_param_grid()

    def _build_param_grid(self) -> List[Dict[str, Any]]:
        """从配置构建参数网格组合

        Returns:
            参数组合列表
        """
        grid = self.config.get("param_grid", {})
        params: List[Dict[str, Any]] = []
        for nleaves in grid.get("num_leaves", [3]):
            for lr in grid.get("learning_rate", [0.003]):
                for alpha in grid.get("reg_alpha", [0]):
                    for lam in grid.get("reg_lambda", [0.05]):
                        params.append({
                            "num_leaves": nleaves,
                            "learning_rate": lr,
                            "reg_alpha": alpha,
                            "reg_lambda": lam,
                        })
        return params

    def prepare_data(
        self,
        series: pd.Series,
        window_size: Optional[int] = None,
        step: Optional[int] = None,
        min_label_count: Optional[int] = None,
    ) -> Tuple[pd.DataFrame, np.ndarray, LabelEncoder, int]:
        """滑动窗口数据准备、特征工程与标签编码

        Args:
            series: 原始数据序列
            window_size: 滑动窗口大小
            step: 滑动步长
            min_label_count: 标签最少出现次数

        Returns:
            (X_features, y_encoded, label_encoder, num_classes)
        """
        window_size = window_size or self.config.get("window_size", 128)
        step = step or FEATURE_CONFIG["window_step"]
        min_label_count = min_label_count or 6

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

        X = calculate_all_features(X)

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
    ) -> "LightGBMModel":
        """训练 LightGBM 模型（自动网格搜索调优）

        Args:
            X_train: 训练特征
            y_train: 训练标签
            X_val: 验证特征
            y_val: 验证标签

        Returns:
            self
        """
        t_start = time.time()

        X_train_split, X_test, y_train_split, y_test = train_test_split(
            X_train,
            y_train,
            test_size=MODEL_CONFIG["test_size"],
            random_state=MODEL_CONFIG["random_state"],
            stratify=y_train,
        )

        num_classes = len(self.label_encoder.classes_) if self.label_encoder else len(np.unique(y_train))

        best_params, best_score, _ = self._grid_search(
            X_train_split, y_train_split, num_classes
        )

        train_data_final = lgb.Dataset(X_train_split, label=y_train_split)

        best_model_params = {
            "objective": "multiclass",
            "num_class": num_classes,
            "metric": "multi_logloss",
            "boosting_type": "gbdt",
            "verbose": -1,
            "seed": MODEL_CONFIG["random_state"],
            **best_params,
        }

        self.model = lgb.train(
            best_model_params,
            train_data_final,
            num_boost_round=self.config.get("boost_round", 500),
        )

        y_pred_proba_train = self.model.predict(X_train_split)
        y_pred_proba_test = self.model.predict(X_test)

        all_classes = np.arange(num_classes)
        train_log_loss = log_loss(y_train_split, y_pred_proba_train, labels=all_classes)
        test_log_loss = log_loss(y_test, y_pred_proba_test, labels=all_classes)

        train_top_k_acc = top_k_accuracy(y_train_split, y_pred_proba_train)
        test_top_k_acc = top_k_accuracy(y_test, y_pred_proba_test)

        train_accuracy = accuracy_score(y_train_split, np.argmax(y_pred_proba_train, axis=1))
        test_accuracy = accuracy_score(y_test, np.argmax(y_pred_proba_test, axis=1))

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
            "overfit_status": overfit_analysis["status"],
            "overfit_severity": overfit_analysis["severity"],
            "overfit_reason": overfit_analysis["reason"],
            "overfit_suggestion": overfit_analysis["suggestion"],
            "elapsed": elapsed,
        }

        self.is_trained = True
        print(f"  ✓ LightGBM 模型训练完成，耗时: {elapsed:.2f}s")
        return self

    def _grid_search(
        self,
        X_train: Union[np.ndarray, pd.DataFrame],
        y_train: Union[np.ndarray, pd.Series],
        num_classes: int,
    ) -> Tuple[Dict[str, Any], float, List[float]]:
        """手动网格搜索超参数调优

        Args:
            X_train: 训练特征
            y_train: 训练标签
            num_classes: 类别数

        Returns:
            (best_params, best_score, all_cv_scores)
        """
        kfold = StratifiedKFold(
            n_splits=3,
            shuffle=True,
            random_state=MODEL_CONFIG["random_state"],
        )

        best_score = -1.0
        best_params = self.config.get("param_grid", {})
        all_cv_scores: List[float] = []

        stop_round = self.config.get("stop_round", 13)
        boost_round = self.config.get("boost_round", 500)

        print(f"  正在网格搜索调参 (共 {len(self._param_grid)} 组参数)...")

        for params in self._param_grid:
            cv_scores = []

            for train_idx, val_idx in kfold.split(X_train, y_train):
                X_tr = X_train.iloc[train_idx] if isinstance(X_train, pd.DataFrame) else X_train[train_idx]
                y_tr = y_train[train_idx]
                X_val = X_train.iloc[val_idx] if isinstance(X_train, pd.DataFrame) else X_train[val_idx]
                y_val = y_train[val_idx]

                train_data = lgb.Dataset(X_tr, label=y_tr)
                val_data = lgb.Dataset(X_val, label=y_val, reference=train_data)

                model_params = {
                    "objective": "multiclass",
                    "num_class": num_classes,
                    "metric": "multi_logloss",
                    "boosting_type": "gbdt",
                    "verbose": -1,
                    "seed": MODEL_CONFIG["random_state"],
                    **params,
                }

                callbacks = [
                    lgb.early_stopping(stopping_rounds=stop_round, verbose=False),
                    lgb.log_evaluation(period=-1),
                ]

                model = lgb.train(
                    model_params,
                    train_data,
                    num_boost_round=boost_round,
                    valid_sets=[val_data],
                    callbacks=callbacks,
                )

                y_pred_proba_val = model.predict(X_val)
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
        proba = self.model.predict(X)
        return np.argmax(proba, axis=1)

    def predict_proba(self, X: Union[np.ndarray, pd.DataFrame]) -> np.ndarray:
        """预测类别概率

        Args:
            X: 输入特征

        Returns:
            预测概率数组，形状 (n_samples, n_classes)
        """
        if not self.is_trained or self.model is None:
            raise RuntimeError("模型尚未训练")
        return self.model.predict(X)

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
            path: 保存路径

        Returns:
            保存的目录路径
        """
        save_dir = Path(path) if path else self._get_save_dir()
        save_dir.mkdir(parents=True, exist_ok=True)

        if self.model is not None:
            self.model.save_model(str(save_dir / "lgb_model.txt"))

        if self.label_encoder is not None:
            joblib.dump(self.label_encoder, save_dir / "label_encoder.joblib")

        if self._X_train_columns is not None:
            joblib.dump(self._X_train_columns, save_dir / "columns.joblib")

        metrics_path = save_dir / "metrics.json"
        with open(metrics_path, "w", encoding="utf-8") as f:
            json.dump(self.metrics, f, ensure_ascii=False, indent=2, default=str)

        print(f"✓ LightGBM 模型已保存至: {save_dir}")
        return save_dir

    def load(self, path: Optional[Union[str, Path]] = None) -> "LightGBMModel":
        """加载模型及标签编码器

        Args:
            path: 加载路径

        Returns:
            self
        """
        load_dir = Path(path) if path else self._get_save_dir()

        model_path = load_dir / "lgb_model.txt"
        encoder_path = load_dir / "label_encoder.joblib"
        columns_path = load_dir / "columns.joblib"

        if model_path.exists():
            self.model = lgb.Booster(model_file=str(model_path))
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
        print(f"✓ LightGBM 模型已从 {load_dir} 加载")
        return self

