"""
双色球预测系统 - 数据加载与预处理模块
提供数据加载、滑动窗口构建、标签编码等核心数据处理功能。
"""
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from pathlib import Path
from typing import List, Tuple

from ml.config import DATA_FILE, FEATURE_CONFIG


def load_data(file_path: str | Path | None = None) -> pd.DataFrame:
    """加载CSV数据文件

    Args:
        file_path: CSV文件路径，为None时使用配置中的默认路径

    Returns:
        加载的DataFrame

    Raises:
        FileNotFoundError: 当文件不存在时
    """
    data_path = Path(file_path) if file_path else DATA_FILE
    if not data_path.exists():
        raise FileNotFoundError(f"数据文件不存在: {data_path}")
    return pd.read_csv(data_path)


def prepare_for_model(
    df: pd.DataFrame,
    target_col: str,
    window_size: int | None = None,
    step: int | None = None,
    min_label_count: int | None = None
) -> Tuple[pd.DataFrame, np.ndarray, LabelEncoder, int]:
    """创建滑动窗口、过滤稀有标签、编码标签

    将单列历史数据通过滑动窗口拆分为特征矩阵和标签，
    过滤掉样本数不足的稀有标签，并使用LabelEncoder进行标签编码。

    Args:
        df: 原始数据DataFrame
        target_col: 目标列名（需要预测的列）
        window_size: 滑动窗口大小，为None时使用配置中的值
        step: 滑动步长，为None时使用配置中的值
        min_label_count: 标签最少出现次数，低于此值的标签将被过滤

    Returns:
        tuple: (X, y_encoded, label_encoder, num_classes)
            - X: 特征DataFrame，形状 (n_samples, window_size)
            - y_encoded: 编码后的标签数组
            - label_encoder: 标签编码器
            - num_classes: 类别数量

    Raises:
        ValueError: 当有效样本数为0时
    """
    window_size = window_size or FEATURE_CONFIG["window_size"]
    step = step or FEATURE_CONFIG["window_step"]
    min_label_count = min_label_count or FEATURE_CONFIG["min_label_count"]

    series = df[target_col]
    n = len(series)
    num_windows = (n - window_size) // step + 1

    if num_windows <= 0:
        raise ValueError(
            f"数据量不足以构建滑动窗口: 数据长度={n}, window_size={window_size}"
        )

    arr = np.asarray(series)
    strides = arr.strides[0]
    new_shape = (num_windows, window_size)
    new_strides = (strides * step, strides)
    windows = np.lib.stride_tricks.as_strided(arr, shape=new_shape, strides=new_strides)

    X = pd.DataFrame(windows)
    last_col = X.columns[-1]
    X = X.rename(columns={last_col: "label"})

    y_raw = X.iloc[:, -1]
    X = X.iloc[:, :-1]

    value_counts = y_raw.value_counts()
    valid_labels = value_counts[value_counts >= min_label_count].index
    valid_mask = y_raw.isin(valid_labels)

    X = X[valid_mask].reset_index(drop=True)
    y_raw = y_raw[valid_mask].reset_index(drop=True)

    if len(X) == 0:
        raise ValueError(
            f"过滤稀有标签后无有效样本: min_label_count={min_label_count}"
        )

    le = LabelEncoder()
    y_encoded = le.fit_transform(y_raw)
    num_classes = len(le.classes_)

    return X, y_encoded, le, num_classes


def extract_feature_columns(df: pd.DataFrame, exclude_cols: List[str] | None = None) -> List[str]:
    """提取特征列，排除非特征列

    Args:
        df: 包含所有列的DataFrame
        exclude_cols: 需要排除的列名列表，为None时使用默认排除列表

    Returns:
        特征列名列表
    """
    if exclude_cols is None:
        exclude_cols = [
            "dNum", "yNum", "mNum", "dDate",
            "Red1", "Red2", "Red3", "Red4", "Red5", "Red6", "Blue1",
            "Sum", "Odd_Count",
        ]

    feature_cols = [col for col in df.columns if col not in exclude_cols]
    return feature_cols


def create_sequential_windows(
    df: pd.DataFrame,
    feature_cols: List[str],
    window_size: int
) -> np.ndarray:
    """创建序列模型的滑动窗口数据集（3D输入: [batch, seq_len, features]）

    Args:
        df: 包含所有特征的DataFrame
        feature_cols: 特征列名列表
        window_size: 时间窗口大小

    Returns:
        np.ndarray: 3D特征数组，形状为 (n_samples, window_size, n_features)
    """
    n = len(df)
    X_data = df[feature_cols].values.astype(np.float32)
    n_features = len(feature_cols)

    X = np.zeros((n - window_size, window_size, n_features), dtype=np.float32)

    for i in range(n - window_size):
        X[i] = X_data[i:i + window_size]

    return X