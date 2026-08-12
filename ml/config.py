"""
双色球预测系统 - 全局配置模块
集中管理所有路径、超参数、模型配置，避免硬编码
"""
from pathlib import Path

# ================= 项目根目录 =================
PROJECT_ROOT = Path(__file__).parent.resolve()

# ================= 数据路径 =================
DATA_DIR = PROJECT_ROOT / "data"
DATA_FILE = DATA_DIR / "1.csv"
MODELS_DIR = PROJECT_ROOT / "saved_models"
OUTPUT_DIR = PROJECT_ROOT / "outputs"

# 确保目录存在
DATA_DIR.mkdir(parents=True, exist_ok=True)
MODELS_DIR.mkdir(parents=True, exist_ok=True)
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# ================= 数据列配置 =================
# 目标列（需要预测的列）
RED_COLS = ['Red1', 'Red2', 'Red3', 'Red4', 'Red5', 'Red6']
BLUE_COLS = ['Blue1']
TARGET_COLS = RED_COLS + BLUE_COLS

# 红球号码范围：1-33
RED_RANGE = range(1, 34)
RED_NUMBERS = 33

# 蓝球号码范围：1-16
BLUE_RANGE = range(1, 17)
BLUE_NUMBERS = 16

# ================= 特征工程配置 =================
FEATURE_CONFIG = {
    # 滑动窗口大小
    "window_size": 165,
    "window_step": 1,
    
    # 频率特征窗口
    "recent_freq_window": 512,
    "poisson_window": 100,
    "entropy_window": 50,
    
    # 统计窗口
    "stat_window": 50,
    "position_window": 30,
    
    # 有效样本过滤阈值（标签最少出现次数）
    "min_label_count": 3,
    
    # 质数集合
    "primes": {2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31},
}

# ================= 通用模型配置 =================
MODEL_CONFIG = {
    # 通用
    "test_size": 0.2,
    "val_size": 0.5,  # 从测试集中划出验证集的比例
    "random_state": 42,
    "top_k": 6,  # 预测前K个号码
    "n_jobs": -1,
}

# ================= 随机森林配置 =================
RF_CONFIG = {
    "n_estimators": [64, 75, 150],
    "max_depth": [4, 7, 12],
    "min_samples_split": [3],
    "min_samples_leaf": [12, 13, 15],
    "base_params": {
        "n_estimators": 50,
        "max_depth": 5,
        "min_samples_split": 20,
        "min_samples_leaf": 10,
        "oob_score": True,
    },
}

# ================= LightGBM 配置 =================
LGB_CONFIG = {
    "boost_round": 500,
    "stop_round": 13,
    "params": {
        "objective": "multiclass",
        "metric": "multi_logloss",
        "boosting_type": "gbdt",
        "verbose": -1,
    },
    "param_grid": {
        "num_leaves": [3, 22, 25, 40, 42],
        "learning_rate": [0.003],
        "reg_alpha": [0],
        "reg_lambda": [0.05],
    },
    "retrain": True,
}

# ================= LSTM 配置 =================
LSTM_CONFIG = {
    "window_size": 128,
    "batch_size": 128,
    "epochs": 256,
    "learning_rate": 0.001,
    "hidden_size": 32,
    "num_layers": 2,
    "dropout_rate": 0.05,
    "early_stop_patience": 7,
    "lr_scheduler_factor": 0.5,
    "lr_scheduler_patience": 3,
    "val_frequency": 10,
    # 蓝球专用
    "blue_hidden_size": 32,
    "blue_num_layers": 2,
    # 红球专用
    "red_hidden_size": 64,
    "red_num_layers": 2,
    "red_dropout": 0.1,
    # 全球模型
    "all_hidden_size": 32,
    "all_num_layers": 1,
    "all_dropout": 0.1,
    "l2_reg": 1e-4,
}

# ================= CNN 配置 =================
CNN_CONFIG = {
    # 数学模型CNN (概率增强)
    "cnn_math": {
        "window_size": 33,
        "window_step": 1,
        "batch_size": 240,
        "epochs": 330,
        "learning_rate": 0.001,
        "conv_out_channels": 5,
        "kernel_size": 9,
        "pool_size": 1,
        "fc_hidden_size": 5,
        "regression_loss_weight": 0.01,
        "dropout_rate": 0.3,
        "weight_decay": 1e-4,
        "entropy_window": 60,
        "markov_states": 7,
        "poisson_window": 330,
        "early_stop_patience": 5,
        "lr_scheduler_factor": 0.9,
        "lr_scheduler_patience": 2,
        "train_log_interval": 20,
        "entropy_chaos_threshold": 4.0,
        "sum_constraint_threshold": 10,
        "chaos_damping_factor": 0.5,
    },
}

# ================= 爬虫配置 =================
SPIDER_CONFIG = {
    "target_url": "https://caipiao.eastmoney.com/pub/Result/History/ssq?page={}",
    "default_save_dir": str(DATA_DIR),
    "csv_filename": "1.csv",
    "request_timeout": 10,
    "request_delay_min": 1,
    "request_delay_max": 3,
    "retry_total": 3,
    "retry_backoff_factor": 1,
    "retry_status_codes": [500, 502, 503, 504],
    "incremental_check_rows": 100,
    "csv_headers": [
        'Deliver Number', 'YearNumber', 'MonthNmb', 'Draw Date',
        'Red1', 'Red2', 'Red3', 'Red4', 'Red5', 'Red6', 'Blue1'
    ],
    "user_agents": [
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36",
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:120.0) Gecko/20100101 Firefox/120.0",
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 10.15; rv:120.0) Gecko/20100101 Firefox/120.0",
    ],
}

# ================= 模型类型枚举 =================
class ModelType:
    RF = "rf"           # 随机森林
    LGBM = "lgbm"       # LightGBM
    LSTM_BLUE = "lstm_blue"
    LSTM_REDS = "lstm_reds"
    LSTM_ALL = "lstm_all"
    CNN_MATH = "cnn_math"

    ALL = [RF, LGBM, LSTM_BLUE, LSTM_REDS, LSTM_ALL, CNN_MATH]


def get_model_config(model_type: str) -> dict:
    """根据模型类型获取对应配置"""
    config_map = {
        ModelType.RF: RF_CONFIG,
        ModelType.LGBM: LGB_CONFIG,
        ModelType.LSTM_BLUE: LSTM_CONFIG,
        ModelType.LSTM_REDS: LSTM_CONFIG,
        ModelType.LSTM_ALL: LSTM_CONFIG,
        ModelType.CNN_MATH: CNN_CONFIG["cnn_math"],
    }
    return config_map.get(model_type, {})