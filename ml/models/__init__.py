"""
双色球预测系统 - 模型模块
导出所有预测模型类，便于统一引用。
所有依赖（lightgbm, torch 等）均为必需依赖，
缺失时直接抛出 ImportError，请通过 `pip install -r requirements.txt` 安装。
"""

# ---- 核心依赖（必装） ----
from ml.models.base_model import BaseModel
from ml.models.rf_model import RandomForestModel
from ml.models.lgb_model import LightGBMModel
from ml.models.lstm_model import LSTMBlueModel, LSTMRedModel, LSTMAllModel
from ml.models.cnn_model import CNNMathModel
from ml.models.set_model import SetRedModel

__all__ = [
    "BaseModel",
    "RandomForestModel",
    "LightGBMModel",
    "LSTMBlueModel",
    "LSTMRedModel",
    "LSTMAllModel",
    "CNNMathModel",
    "SetRedModel",
]
