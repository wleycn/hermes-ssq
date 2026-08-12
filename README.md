# 双色球预测系统 (SSQ)

基于机器学习与深度学习的双色球彩票号码预测系统。通过分析历史开奖数据，使用多种模型（随机森林、LightGBM、LSTM、CNN）预测下一期可能出现的号码及其概率。

## 项目结构

```
SSQ/
├── .venv/                        # Python 虚拟环境（不入版本控制）
├── .vscode/settings.json         # IDE 配置（解释器指向 .venv）
├── .gitignore                    # Git 忽略规则
├── requirements.txt              # Python 依赖清单
├── README.md                     # 本文件
└── ml/                           # 源代码
    ├── __init__.py
    ├── main.py                   # 主入口脚本（训练+预测）
    ├── config.py                 # 全局配置（路径、超参数、模型类型）
    ├── data/                     # 数据模块
    │   ├── __init__.py
    │   ├── dataset.py            # 数据加载与预处理
    │   └── spider.py             # 历史数据爬虫（东方财富网）
    ├── features/
    │   ├── __init__.py
    │   └── feature_engineer.py   # 特征工程（统计/频率/熵/马尔可夫等）
    ├── models/                   # 模型实现
    │   ├── __init__.py           # 模型导出（懒加载）
    │   ├── base_model.py         # 抽象基类 BaseModel
    │   ├── rf_model.py           # 随机森林模型
    │   ├── lgb_model.py          # LightGBM 模型
    │   ├── lstm_model.py         # LSTM 模型（蓝球/红球/全球）
    │   └── cnn_model.py          # CNN 数学增强模型
    ├── utils/
    │   ├── __init__.py
    │   └── helpers.py            # 通用工具函数
    ├── data/1.csv                # 原始数据（3485 期开奖记录）
    ├── saved_models/             # 训练好的模型文件（不入版本控制）
    ├── outputs/                  # 预测结果 CSV（不入版本控制）
    ├── logs/                     # 运行日志（按天生成，不入版本控制）
    └── legacy/                   # 早期探索代码（已归档，不参与生产）
```

## 环境搭建

### 1. 创建虚拟环境

```bash
# 使用 Python 3.13+ 创建虚拟环境
python -m venv .venv

# 激活虚拟环境
.venv\Scripts\Activate.ps1        # PowerShell
# .venv\Scripts\activate.bat      # CMD
# source .venv/bin/activate       # Linux/macOS
```

### 2. 安装依赖

```bash
pip install -r requirements.txt
```

依赖清单（10 个包）：

| 包 | 版本 | 用途 |
|---|---|---|
| numpy | 2.4.6 | 数值计算 |
| pandas | 3.0.3 | 数据处理 |
| scipy | 1.18.0 | 统计分布（泊松/正态） |
| scikit-learn | 1.9.0 | 随机森林、数据划分 |
| lightgbm | 4.7.0 | LightGBM 模型 |
| torch | 2.13.0+cpu | LSTM/CNN 深度学习 |
| joblib | 1.5.3 | 模型序列化 |
| requests | 2.34.2 | 数据爬虫 |
| urllib3 | 2.7.0 | HTTP 重试策略 |
| lxml | >=5.0.0 | HTML 解析 |

## 使用方法

### 支持的命令

系统仅保留 2 个核心命令，覆盖 6 种模型：

| 命令 | 适用模型 | 说明 |
|------|---------|------|
| `train-predict` | LSTM 系列、CNN | 训练后自动预测 |
| `train-predict-batch` | RF、LGBM | 批量训练+预测所有红球和蓝球列 |

### 运行示例

```bash
# 激活虚拟环境后执行

# 1. 随机森林：批量训练+预测所有红球+蓝球
python -m ml.main train-predict-batch --model rf --columns all

# 2. LightGBM：批量训练+预测所有红球+蓝球
python -m ml.main train-predict-batch --model lgbm --columns all

# 3. LSTM 全球模型：训练+预测红球+蓝球
python -m ml.main train-predict --model lstm_all

# 4. LSTM 蓝球模型：训练+预测蓝球
python -m ml.main train-predict --model lstm_blue

# 5. LSTM 红球模型：训练+预测红球
python -m ml.main train-predict --model lstm_reds

# 6. CNN 数学增强模型：训练+预测红球+蓝球
python -m ml.main train-predict --model cnn_math
```

### 命令参数

```
train-predict:
  --model, -m      模型类型（lstm_all/lstm_blue/lstm_reds/cnn_math）
  --data, -d       数据文件路径（默认 ml/data/1.csv）
  --retrain, -r    是否重新训练（Y/N，默认 Y）

train-predict-batch:
  --model, -m      模型类型（rf/lgbm）
  --columns        列集合（red=红球 / blue=蓝球 / all=红+蓝，默认 red）
  --data, -d       数据文件路径（默认 ml/data/1.csv）
  --retrain, -r    是否重新训练（Y/N，默认 Y）
```

## 模型说明

| 模型 | 类型 | 预测目标 | 核心算法 |
|------|------|---------|---------|
| RF | 传统 ML | 红球各位置 / 蓝球 | 随机森林分类 |
| LGBM | 传统 ML | 红球各位置 / 蓝球 | LightGBM 梯度提升 |
| LSTM_BLUE | 深度学习 | 蓝球 1-16 | LSTM 二分类 |
| LSTM_REDS | 深度学习 | 红球 1-33 | LSTM 多标签分类 |
| LSTM_ALL | 深度学习 | 红球+蓝球联合 | LSTM 多任务学习 |
| CNN_MATH | 深度学习 | 红球+蓝球联合 | CNN + 数学后处理 |

### CNN_MATH 后处理流程

CNN 数学增强模型在神经网络输出后，依次执行 4 步数学约束：

1. **信息熵检查**：检测预测是否处于高混乱状态，若是则做平滑处理
2. **正态分布过滤**：预测和值超出 μ±2.58σ 时替换极端号码
3. **泊松分布优化**：根据泊松概率调整号码选择
4. **和值约束**：用回归预测的 Next_Sum 校正最终和值

## 输出说明

### 预测结果 CSV

文件路径：`ml/outputs/prediction_<model>_<timestamp>.csv`

格式：

```csv
ModelType,BallType,BallNumber,Prob
lstm_blue,blue,15,0.098722
lstm_blue,blue,7,0.085432
...
```

### 批量预测汇总 CSV

文件路径：`ml/outputs/prediction_summary_<model>_<timestamp>.csv`

汇总 RF/LGBM 所有列的 Top-K 预测结果。

### 运行日志

文件路径：`ml/logs/ssq_main_<YYYYMMDD>.log`（按天生成，同一天多次运行追加到同一文件）

## 数据采集

如需更新历史数据，运行爬虫模块：

```bash
python -m ml.data.spider
```

数据来源：东方财富网，自动爬取并增量保存到 `ml/data/1.csv`。

## 技术栈

- **语言**：Python 3.13+
- **传统 ML**：scikit-learn、LightGBM
- **深度学习**：PyTorch（CPU 版）
- **数据处理**：pandas、numpy、scipy
- **数据采集**：requests、lxml

## 注意事项

- 本系统仅供技术学习与研究，彩票中奖为随机事件，预测结果不构成任何投注建议
- `ml/legacy/` 目录为早期探索代码，已归档隔离，不参与生产流程
- 模型文件、预测结果、日志均不入版本控制（见 `.gitignore`）
