# PROJECT.md — SSQ（双色球预测）

> 编码基础设置层（与 AGENTS.md 通用技能层分离）。Cheese 动手前先读此文件。

## 技术栈
- Language: Python 3.14（注意：PEP 668，装依赖须 `.venv` 或 uv）
- 数据：PostgreSQL **`hermes` 库，schema `ssq`**（表 `ssq.draw_history` 全量开奖 / `ssq.model_predictions` 模型概率(保留30天) / `ssq.draw_stats` 奖金销量奖池）
- 数据源：**中彩网(权威,需特定 Referer) + 东方财富 + 网易 多源轮询**；**无 akshare**（akshare 是股票/基金行情库，与开奖数据无关，勿用）
- 模型：scikit-learn + PyTorch；模型清单以 `batch_predict_pg.MODELS` 为单一真相源（当前：rf / lightgbm / cnn_reg / lstm / transformer / cdm 等）
- 验证：pytest（`.pytest_cache` 已存在）

## 目录分层（编码前先定位）
```
SSQ/
├── ml/                 # 模型训练/预测
│   ├── data/           # 抓取/入库 (update_ssq/append_ssq + db_draw.py PG 业务写)
│   ├── models/         # 各模型实现 (rf/lightgbm/cnn/lstm/transformer/cdm)
│   ├── features/       # 特征工程
│   ├── conformal/      # conformal 预测区间
│   ├── probes/         # 随机性检验探针 (spectral 等)
│   └── eval/           # 回测
├── analysis/           # 统计检验 / EV 分析 (ssq_ev.py 等)
├── research/           # 研究方法论 (ARCHITECTURE.md / reports/ / backfill 脚本)  ← 架构权威文档在此
├── _verify/            # 验证脚本
├── docs/               # 其他研究资料 (incremental_validation / steiner_walkforward / feature_audit)
└── *.py                # 顶层入口: batch_predict_pg / retrain_pipeline / select_numbers / ssq_send_picks / cleanup_predictions / pg_schema
```
- **分层纪律**：顶层 `*.py` 是入口；`ml/` 与 `analysis/` 不互相循环依赖；改 `pg_schema` 必查下游。

## 关键约定
- **DB 三层入口**：连接统一走 `ml/pg_conn.py`（读 `.env` 的 `DATABASE_URL`）；schema 定义/校验在 `pg_schema.py`；业务 upsert 在 `ml/data/db_draw.py`。禁止业务脚本硬编码表名/列名或连接串。
- 预测结果落 PG 须带 **`run_at / model / ball_type / num` 四键**（红蓝分别、每模型独立），便于回测与 UPSERT（`ON CONFLICT DO UPDATE`）。
- 类型注解必写（你的偏好：代码可信靠类型）。

## 验证命令
> venv 绝对路径：`/home/hermes/workspace/python/SSQ/.venv/bin`（ruff / pytest / mypy 已装）
- 语法：`<venv>/python -c "import ast; ast.parse(open('file').read())"`
- lint：`<venv>/ruff check <file_or_dir>`（改装后必跑，修真问题不盲忽略）
- 类型：`python -m mypy <module>`（须在 `.venv` 内）
- 单测：`<venv>/python -m pytest _verify/ -q`
- 回测证据：改模型后跑 `evaluate.py` 对比基线，不允许"应该更好"
- **动手前先过文末「编码架构前置闸」4 问**（分层/契约/失败/验证点），无 arch note 不写码

## 已知坑（踩过）
- 预测 `evaluate.py` 51KB，改前先读全文件，别凭记忆改
- PG 连接走 `.env` 的 `DATABASE_URL`，不硬编码
- `is not None` 对未定义变量恒真（Undefined≠None）——判"是否传入"用 `is defined`（Jinja/模板语境）

## 编码架构前置闸（每次动手前答）
1. 分层：动哪个层？影响 `ml/` 还是 `analysis/`？
2. 契约：输入输出确切字段？空/异常期号怎么处理？
3. 失败模式：预测落库失败谁报警？静默覆盖旧结果？
4. 验证点：最小可验证增量？跑哪条 pytest 证明活着？
