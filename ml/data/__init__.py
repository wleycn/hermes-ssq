"""
双色球预测系统 - 数据模块
包含数据爬虫、数据加载与预处理功能。
所有依赖（requests, lxml 等）均为必需依赖，
缺失时直接抛出 ImportError，请通过 `pip install -r requirements.txt` 安装。
"""
from ml.data.dataset import (
    load_data,
    prepare_for_model,
    extract_feature_columns,
    create_sequential_windows,
)
from ml.data.spider import SsqSpider

__all__ = [
    "SsqSpider",
    "load_data",
    "prepare_for_model",
    "extract_feature_columns",
    "create_sequential_windows",
]