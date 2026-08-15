# 双色球预测系统 (SSQ)

基于机器学习与深度学习的双色球彩票号码预测系统。通过分析历史开奖数据，使用多种模型（随机森林、LightGBM、LSTM、CNN）预测下一期可能出现的号码及其概率，并将各模型概率集成后生成候选号码、通过邮件推送。

> ⚠️ 本系统仅供技术学习与研究，彩票中奖为随机事件，预测结果不构成任何投注建议。
>
> 🔬 **诚实结论（历史回测坐实）**：双色球为独立均匀随机过程，任何选号策略的期望收益 < 成本。wheel 30 注 ROI 历史回测约 **-70% ~ -75%**（长期必亏）。本系统的价值是"统计严谨 + 诚实检验 + 自动化流程"，不是"能赚钱"。

## 项目结构

```
SSQ/
├── .venv/                        # Python 虚拟环境（不入版本控制）
├── .git/ .gitee/                 # git 双远程（github + gitee）
├── .gitignore                    # Git 忽略规则
├── requirements.txt              # Python 依赖清单
├── README.md                     # 本文件
├── batch_predict_pg.py           # 批量训练 8 模型 → 概率写入 PostgreSQL
├── pg_schema.py                  # PG 建表 + 1.csv 导入 draw_history + data_date 列迁移
├── cleanup_predictions.py        # 预测表 30 天滚动清理（仅动了 model_predictions）
├── select_numbers.py             # 读 PG 集成概率 → 生成 5 组候选号码（等权/EBMA 集成）
├── send_ssq_picks.py             # 组装中文邮件正文 → smtplib 直发（旧入口，保留）
├── retrain_pipeline.py           # 一键重训+验证+发邮件（封装 batch_predict_pg + 外部脚本）
├── wheel.py                      # 旋转矩阵覆盖设计（贪心 4-子集覆盖）
├── _verify/                      # pytest 验证套件
├── analysis/                     # 历史分析与探索脚本
│   ├── wheel_ledger.py           # wheel ROI 模拟账本（口径 B+C 实跑）
│   └── pool_compare_backtest.py  # ML池 vs 随机池 + wheel 历史回测对比
└── ml/                           # 源代码
    ├── __init__.py
    ├── main.py                   # 模型训练+预测入口（单模型）
    ├── config.py                 # 全局配置（路径、超参数、模型类型）
    ├── ensemble.py               # EBMA 轻量融合模块（零依赖，softmax of 历史 log-likelihood）
    ├── popularity.py             # 冷号加权（6 规则, λ=0.3, 5 组模式用）
    ├── spectral.py               # 蓝球 3 门随机性测试器（21 个函数）
    ├── spectral_red.py           # 红球 3 路径随机性测试器
    ├── spectral_chaos.py         # 混沌/相空间重构检验器（Takens+Lyapunov+替代检验）
    ├── data/
    │   ├── dataset.py            # 数据加载与预处理
    │   ├── spider.py             # 历史数据爬虫（东方财富网，备用）
    │   ├── update_ssq.py         # 三源抓取更新主入口（中彩网+EastMoney+网易）
    │   └── append_ssq.py         # 幂等追加单行到 1.csv（CRLF 保真）
    ├── features/feature_engineer.py  # 特征工程（统计/频率/熵/马尔可夫等）
    ├── models/                   # 模型实现（rf/lgbm/lstm/cnn/transformer/cdm）
    ├── utils/helpers.py          # 通用工具函数
    ├── data/1.csv                # 原始数据（3489 期开奖记录，CRLF）
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
1.csv (3489期历史)
   │  update_ssq.py 三源抓取更新（中彩网官方API + EastMoney + 网易交叉校验）
   │  ⚠️ 仅在开奖日（周二/四/日 22:00）由 cron 自动触发
   ▼
ml.main 训练 8 模型 (rf / lgbm / cnn_math / lstm_blue / lstm_reds / lstm_all / transformer_all / cdm)
   │  batch_predict_pg.py 批量预测 → 写入 PG (每月1号 03:00 cron 自动重训)
   ▼
PostgreSQL ssq.model_predictions (每模型每球种独立取最新 run_at)
   │  select_numbers.py 读最新 → 集成（等权均值 or EBMA softmax 加权）
   ├──────────────────────────────┐
   ▼                              ▼
5 组候选号码                   wheel 旋转矩阵
（受控随机加权+奇偶/         （Top-18球池, 30注,
大小比约束+冷热加权）          4-子集覆盖, pass_rate 95.4%）
   │                              │
   └──────────── 合并 ─────────────┘
                 │
                 ▼
        邮件推送（开奖日 22:15 cron 触发 ssq_send_picks.py）
        - 5组(按流行度降序) + wheel 30注
        - 主题：双色球XXXXXX期推荐: 5组 + wheel30注(通过率95.4%)
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
# 训练+预测 8 模型，概率写入 PostgreSQL（schema: ssq）
.venv/bin/python batch_predict_pg.py
```

每次运行产生一个批次（`run_at` 时间戳 + `data_date` 北京日期），8 模型各对 33 红球 / 16 蓝球输出概率，全量写入 `ssq.model_predictions`。

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

通过 Hermes cron 配置，三个任务全部 local-only（结果存文件，不重复发通知，脚本自带 163 邮件）：

| 任务 | 时间 | 命令 | 作用 |
|------|------|------|------|
| 抓开奖 | 开奖日(二/四/日) 22:00 | `python ml/data/update_ssq.py` | 三源抓取+入库+发开奖邮件 |
| 发下期预测 | 开奖日(二/四/日) 22:15 | `python ~/.hermes/scripts/ssq_send_picks.py` | 自动算下期+生成推荐+发邮件 |
| 月度重训 | 每月1号 03:00 | `python retrain_pipeline.py --no-email` | 重训8模型(不发邮件, 发预测自动用新概率) |

> 抓开奖与发预测间隔 15 分钟（22:00→22:15），确保本期已入库后再生成下期预测。
> 月度重训：Rocky 要求"模型一个月重训一次"，重训后 load_latest_probs 自动取每模型最新概率。

**首次真实运行**：2026-08-16（周日）22:00/22:15。

## 模型说明

| 模型 | 类型 | 预测目标 | 核心算法 |
|------|------|---------|---------|
| RF | 传统 ML | 红球各位置 / 蓝球 | 随机森林分类（逐位置聚合为 33红/16蓝全量概率） |
| LGBM | 传统 ML | 红球各位置 / 蓝球 | LightGBM 梯度提升（同上聚合） |
| LSTM_BLUE | 深度学习 | 蓝球 1-16 | LSTM 二分类（仅蓝球） |
| LSTM_REDS | 深度学习 | 红球 1-33 | LSTM 多标签分类（仅红球） |
| LSTM_ALL | 深度学习 | 红球+蓝球联合 | LSTM 多任务学习 |
| CNN_MATH | 深度学习 | 红球+蓝球联合 | CNN + 数学后处理 |
| TRANSFORMER_ALL | 深度学习 | 红球+蓝球联合 | Transformer 编码器（自注意力，窗口序列） |
| CDM | 贝叶斯统计 | 红球+蓝球联合 | Compound-Dirichlet-Multinomial 后验均值（频数+先验平滑） |

### 多模型集成（select_numbers.py）

- **等权均值**（默认）：8 模型概率简单平均，向后兼容
- **EBMA**（`--ensemble ebma`）：按历史开奖 log-likelihood softmax 加权，默认 tau=8000（接近等权不坍缩，因随机过程上模型差异属噪声）
- **取数逻辑**：每模型每球种独立取最新 `run_at`（非单一全局最新），避免"必须一次跑完所有模型共享 run_at"的耦合

### 混沌 / 相空间重构检验（ml/spectral_chaos.py）

Takens 延迟嵌入 + Rosenstein Lyapunov + FFT 相位随机化替代检验 + 样本熵：
- **定位**：随机性检验器（非预测器），延续光谱探针的"检验优先"思路
- **实测结论**：SSQ 开奖序列无混沌结构（Lyapunov z=0.29, p=0.78，打乱排序后）
- **重要陷阱（已记录）**：开奖号按升序排列（Red1<...<Red6）拍平后会引入**排序伪影**——未打乱时 Lyapunov 误报 CHAOTIC（z=9.88），每期内打乱后回到 RANDOM。**任何序列化开奖数据做时序分析前必须先打破排序**
- **样本熵 FFT 分支有已知假阳性**（纯随机整数序列误报 CHAOTIC z=50.7），需以 Lyapunov 分支为准

### 马尔可夫马氏性闸门（feature_engineer.py）

新增卡方列联表马氏性检验（α=0.05）：
- 实测奇偶比序列 p=0.7494 >> 0.05，不通过
- 闸门意义：在随机序列上，马尔可夫预测无意义，下游自动 markov_valid=False

### 旋转矩阵覆盖设计（wheel.py）

贪心 4-子集覆盖算法：
- 池大小 N（默认 18），目标：30 注内让尽可能多的 6-子集"至少 4 红命中"
- N=18 时 pass_rate=95.42%（穷举精确计算 C(18,6)=18564 种）
- **诚实事实**：覆盖率只依赖池大小，与具体哪 18 个球无关。ML Top-18 池与随机 18 球池期望等价（历史回测验证：ML ROI -75.43% vs 随机中位数 -75.87%，差异 < 1 标准差）

### 模型加速（实测）

LSTM 在 CPU 上原训练极慢（lstm_all 窗口 330 + 256 epoch ≈ 22 分钟）。已调优：

- `lstm_all` 窗口 `330 → 128`
- 全局 `epochs 256 → 80`（早停 patience=7）
- 训练循环加进度打印（消除"像卡死"误判）

加速后：lstm_all 单模型训练 **~59s**，transformer_all 约 9 分钟（8 模型含 transformer/cdm 批量入库约 15 分钟）。

Transformer 性能优化（2026-08-15）：注意力 O(n²)，初始 window=128 + 每 epoch 验证导致 **32 分钟**。已调优：

- `window_size 128 → 32`（注意力计算省 16 倍）
- `batch_size 64 → 128`、`val_frequency 5 → 10`（对齐 LSTM）
- `dim_feedforward 256 → 128`（对齐模型内部默认）

调优后：**32 分钟 → 9 分钟**（3.6 倍）。随后发现真正瓶颈并非注意力/维度，而是 **CPU 线程数**：PyTorch 默认 8 线程跑 64 维小张量时，线程调度开销吞掉计算收益（单 Linear 8 线程 12.6ms vs 4 线程 0.19ms，慢 66 倍）。`TransformerAllModel.train()` 内显式 `torch.set_num_threads(4)` 后，实测单模型重训 **32 分钟 → 29 秒（18.8 倍）**，8 模型批量约 10 分钟内完成。4 线程为本机最优（实测 1/2/4/8 线程对比）。

⚠️ **2026-08-15 多agent独立验收修正（重要）**：早先结论"仅 transformer 需要、LSTM/CNN 免疫"是**错的**——测量被 `torch.set_num_threads` 的**进程级全局性**污染（在 transformer train() 内设置后，同进程内 LSTM/CNN 测量也变成 4 线程）。独立 test 子代理在干净进程逐模型扫描实测：**所有 torch 模型在 8 线程下都崩溃级慢**——LSTM +8818%、CNN +2860%、Transformer +591%（8 vs 4 线程）。已修复：`batch_predict_pg.py` **入口统一 `torch.set_num_threads(4)`**（不能在模型内部设，MODELS 里 LSTM/CNN 排在 transformer 前面，会先用 8 线程跑完）。

⚠️ 性能教训：CPU 上小模型训练务必实测线程数，默认 8 线程对 64 维小张量是灾难；且 `set_num_threads` 是全局的，任何"某模型免疫"的结论必须在新进程、干净环境验证。

⚠️ 配置注意：`ml/config.py` 的 `TRANSFORMER_CONFIG` 是唯一生效配置，`ml/main.py` 实例化时显式传入；`transformer_model.py` 模块内默认值仅作 fallback。

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
- 历史回测坐实：双色球为独立均匀随机过程，wheel 30 注 ROI 约 **-70% ~ -75%**（长期必亏）。请勿投入超出承受能力的资金
- `ml/legacy/` 目录为早期探索代码，已归档隔离，不参与生产流程
- 模型文件、预测结果、日志均不入版本控制（见 `.gitignore`）
- `data_date` 写入使用 Python 侧北京日期（`datetime.now().date()`），不依赖 PG 的 `CURRENT_DATE`（服务器时区为 Etc/UTC，差 8 小时）
- 预测表清理为不可逆操作，默认 dry-run，须显式 `--confirm` 才执行
- **脚本 `~/.hermes/scripts/ssq_send_picks.py` 含数据库和邮件凭证，在 `~/.hermes/scripts/` 目录（不在 git 跟踪范围），切勿复制到仓库**
