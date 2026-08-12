# 双色球预测系统 (SSQ)

基于机器学习与深度学习的双色球彩票号码预测系统。通过分析历史开奖数据，使用多种模型（随机森林、LightGBM、LSTM、CNN）预测下一期可能出现的号码及其概率，并将各模型概率集成后生成候选号码、通过邮件推送。

> ⚠️ 本系统仅供技术学习与研究，彩票中奖为随机事件，预测结果不构成任何投注建议。

## 项目结构

```
SSQ/
├── .venv/                        # Python 虚拟环境（不入版本控制）
├── .git/ .gitee/                 # git 双远程（github + gitee）
├── .gitignore                    # Git 忽略规则
├── requirements.txt              # Python 依赖清单
├── README.md                     # 本文件
├── batch_predict_pg.py           # 批量训练 6 模型 → 概率写入 PostgreSQL
├── pg_schema.py                  # PG 建表 + 1.csv 导入 draw_history + data_date 列迁移
├── cleanup_predictions.py        # 预测表 30 天滚动清理（仅动了 model_predictions）
├── select_numbers.py             # 读 PG 集成概率 → 生成 5 组候选号码
├── send_ssq_picks.py             # 组装中文邮件正文 → smtplib 直发 wleycn@163.com
├── _verify/                      # pytest 验证套件（27 passed）
├── analysis/                     # 历史分析与探索脚本
└── ml/                           # 源代码
    ├── __init__.py
    ├── main.py                   # 模型训练+预测入口（单模型）
    ├── config.py                 # 全局配置（路径、超参数、模型类型）
    ├── data/
    │   ├── dataset.py            # 数据加载与预处理
    │   ├── spider.py             # 历史数据爬虫（东方财富网，备用）
    │   ├── update_ssq.py         # 三源抓取更新主入口（中彩网+EastMoney+网易）
    │   └── append_ssq.py         # 幂等追加单行到 1.csv（CRLF 保真）
    ├── features/feature_engineer.py  # 特征工程（统计/频率/熵/马尔可夫等）
    ├── models/                   # 模型实现（rf/lgbm/lstm/cnn）
    ├── utils/helpers.py          # 通用工具函数
    ├── data/1.csv                # 原始数据（3488 期开奖记录，CRLF）
    ├── saved_models/             # 训练好的模型文件（不入版本控制）
    ├── outputs/                  # 预测结果 CSV（不入版本控制）
    └── legacy/                   # 早期探索代码（已归档，不参与生产）
```

## 环境搭建

### 1. 创建虚拟环境

```bash
python -m venv .venv
source .venv/bin/activate       # Linux/macOS
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
| torch | 2.13.0+cpu | LSTM/CNN 深度学习（CPU 版） |
| joblib | 1.5.3 | 模型序列化 |
| requests | 2.34.2 | 数据爬虫 |
| urllib3 | 2.7.0 | HTTP 重试策略 |
| lxml | >=5.0.0 | HTML 解析 |
| psycopg | 3.3.4 | PostgreSQL 驱动（批量入库/查询） |

## 核心流程

```
1.csv (3488期历史)
   │  update_ssq.py 三源抓取更新（中彩网官方API + EastMoney + 网易交叉校验）
   ▼
ml.main 训练 6 模型 (rf / lgbm / cnn_math / lstm_blue / lstm_reds / lstm_all)
   │  batch_predict_pg.py 批量预测 → 写入 PG
   ▼
PostgreSQL ssq.model_predictions (data_date 列, 每次预测存北京当天)
   │  select_numbers.py 读最新批次 → 各模型概率均值集成
   ▼
5 组候选号码（红球受控随机加权 + 奇偶/大小比约束 + 热号标注）
   │  send_ssq_picks.py 组装中文邮件
   ▼
smtp.163.com → wleycn@163.com
```

### 1. 数据更新（三源）

```bash
# 主入口：中彩网官方 API 为主源，EastMoney + 网易交叉校验
.venv/bin/python ml/data/update_ssq.py

# 仅抓取不发送邮件
.venv/bin/python ml/data/update_ssq.py --no-email
```

数据写入 `ml/data/1.csv`（CRLF 行尾，列序 `dNum,yNum,mNum,dDate,Red1..Red6,Blue1`，两位零填充）。尾部幂等查重，重复运行不污染。

### 2. 批量预测入库

```bash
# 训练+预测 6 模型，概率写入 PostgreSQL（schema: ssq）
.venv/bin/python batch_predict_pg.py
```

每次运行产生一个批次（`run_at` 时间戳 + `data_date` 北京日期），6 模型各对 33 红球 / 16 蓝球输出概率，全量写入 `ssq.model_predictions`。

### 3. 选号 + 邮件推送

```bash
.venv/bin/python send_ssq_picks.py            # 生成并发送邮件
.venv/bin/python send_ssq_picks.py --dry-run  # 只打印正文不发送
.venv/bin/python send_ssq_picks.py --seed 7   # 换随机种子重生成
```

邮件含：5 组候选号码、热号标注（集成概率前 8）、完整选取逻辑说明。

### 4. PostgreSQL 存储层

数据库：本地 docker PostgreSQL（`127.0.0.1:5432`，库 `hermes`，schema `ssq`）。

| 表 | 用途 | 清理策略 |
|----|------|---------|
| `ssq.draw_history` | 1.csv 全量镜像归档（开奖历史） | **全量保留，永不清理** |
| `ssq.model_predictions` | 各模型红/蓝球预测概率 | **滚动 30 自然日清理**（见下） |

建表与导入：

```bash
.venv/bin/python pg_schema.py    # 幂等：建表 + 导入 draw_history(3488行) + 加 data_date 列
```

`data_date` 列类型为 `DATE NOT NULL`，每次预测写入北京当天日期（规避 PG 会话时区 Etc/UTC 偏差）。

清理（仅作用于预测表，不可逆）：

```bash
.venv/bin/python cleanup_predictions.py        # 默认 dry-run：打印将删计数，不删
.venv/bin/python cleanup_predictions.py --confirm   # 真实删除 data_date < 今天-30天 的行
```

> 开奖历史表 `draw_history` 硬编码排除在清理范围外。1.csv 经 update_ssq 更新后，需重跑 `pg_schema.py` 同步（函数幂等）。

### 5. 自动化（cron）

通过 Hermes cron 配置：`0 22 * * 2,4,0`（**仅双色球开奖日周二/四/日 22:00**）触发 `update_ssq.py`，有更新则抓取入库并邮件通知。

## 模型说明

| 模型 | 类型 | 预测目标 | 核心算法 |
|------|------|---------|---------|
| RF | 传统 ML | 红球各位置 / 蓝球 | 随机森林分类（逐位置聚合为 33红/16蓝全量概率） |
| LGBM | 传统 ML | 红球各位置 / 蓝球 | LightGBM 梯度提升（同上聚合） |
| LSTM_BLUE | 深度学习 | 蓝球 1-16 | LSTM 二分类（仅蓝球） |
| LSTM_REDS | 深度学习 | 红球 1-33 | LSTM 多标签分类（仅红球） |
| LSTM_ALL | 深度学习 | 红球+蓝球联合 | LSTM 多任务学习 |
| CNN_MATH | 深度学习 | 红球+蓝球联合 | CNN + 数学后处理 |

### 模型加速（实测）

LSTM 在 CPU 上原训练极慢（lstm_all 窗口 330 + 256 epoch ≈ 22 分钟）。已调优：

- `lstm_all` 窗口 `330 → 128`
- 全局 `epochs 256 → 80`（早停 patience=7）
- 训练循环加进度打印（消除"像卡死"误判）

加速后：lstm_all 单模型训练 **~59s**，全 6 模型批量入库约 12 分钟。

### CNN_MATH 后处理流程

CNN 数学增强模型在神经网络输出后，依次执行 4 步数学约束：

1. **信息熵检查**：检测预测是否处于高混乱状态，若是则做平滑处理
2. **正态分布过滤**：预测和值超出 μ±2.58σ 时替换极端号码
3. **泊松分布优化**：根据泊松概率调整号码选择
4. **和值约束**：用回归预测的 Next_Sum 校正最终和值

## 使用方法（单模型训练/预测）

如需单独训练某个模型（调试用），仍可用 `ml.main`：

```bash
python -m ml.main train-predict-batch --model rf --columns all
python -m ml.main train-predict-batch --model lgbm --columns all
python -m ml.main train-predict --model lstm_all
python -m ml.main train-predict --model lstm_blue
python -m ml.main train-predict --model lstm_reds
python -m ml.main train-predict --model cnn_math
```

`train-predict` 支持 `--data`（默认 ml/data/1.csv）、`--retrain`（Y/N）；`train-predict-batch` 支持 `--columns`（red/blue/all）。

## 输出说明

### 预测结果 CSV

文件路径：`ml/outputs/prediction_<model>_<timestamp>.csv`

```csv
ModelType,BallType,BallNumber,Prob
lstm_blue,blue,15,0.098722
lstm_blue,blue,7,0.085432
...
```

### 运行日志

文件路径：`ml/logs/ssq_main_<YYYYMMDD>.log`（按天生成，同一天多次运行追加到同一文件）

## 技术栈

- **语言**：Python 3.13+
- **传统 ML**：scikit-learn、LightGBM
- **深度学习**：PyTorch（CPU 版）
- **数据处理**：pandas、numpy、scipy
- **数据存储**：PostgreSQL + psycopg
- **数据采集**：requests、lxml
- **邮件推送**：smtplib（smtp.163.com:465）

## 注意事项

- 本系统仅供技术学习与研究，彩票中奖为随机事件，预测结果不构成任何投注建议
- `ml/legacy/` 目录为早期探索代码，已归档隔离，不参与生产流程
- 模型文件、预测结果、日志均不入版本控制（见 `.gitignore`）
- `data_date` 写入使用 Python 侧北京日期（`datetime.now().date()`），不依赖 PG 的 `CURRENT_DATE`（服务器时区为 Etc/UTC，差 8 小时）
- 预测表清理为不可逆操作，默认 dry-run，须显式 `--confirm` 才执行
