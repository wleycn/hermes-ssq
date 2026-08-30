# T13 特征增量验证报告（incremental validation）

> 验证人：TestAutomationEngineer（独立验证，不依赖 Dev 自报）
> 执行时间：2026-08-13 09:36:56（`run_at`，evaluate.py 落档）
> 命令：`.venv/bin/python evaluate.py --features ac,entropy,hot_cold,crf,diversity --horizon 100 --train-min 800 --out /home/hermes/workspace/data-center/ssq/analysis/results/feature_validation.json`
> 产物：`/home/hermes/workspace/data-center/ssq/analysis/results/feature_validation.json`（全量，n_trials=1000，seed=1，pool_size=12）
> 基线对照：`/home/hermes/workspace/data-center/ssq/analysis/results/validation_baseline_random.json`（random 策略）、`/home/hermes/workspace/data-center/ssq/analysis/results/validation_baseline_freq.json`（freq 策略）

## 判定规则（PRD Q3 裁定）

**红球 mean_hits > 基线均值 + 1.96×SE 且 n ≥ 50 → keep（保留特征开关）；否则 rollback（回退关闭）。**

- MC 随机基线（N=1000/期，100 期）：**baseline_mean = 1.0894**（理论期望 6×6/33 = 1.0909，偏差 -0.0015，抽样误差内）
- 各特征 95% CI 与显著性阈值（基线 + 1.96×SE）见下表。

## 汇总表

| 特征 | n | 红球 mean_hits | 基线均值 (N=1000 MC) | Δ | Δ 95% CI | t 检验 p 值 | 阈值 (基线+1.96SE) | n≥50 | 显著 | 判定 |
|---|---:|---:|---:|---:|---|---:|---|---:|---|
| ac | 100 | 1.0800 | 1.0894 | -0.0094 | [-0.2393, +0.2205] | 0.9084 | 1.2488 | ✓ | ✗ | **rollback** |
| entropy | 100 | 1.1100 | 1.0894 | +0.0206 | [-0.1612, +0.2024] | 0.8209 | 1.2676 | ✓ | ✗ | **rollback** |
| hot_cold | 100 | 1.1000 | 1.0894 | +0.0106 | [-0.1488, +0.1700] | 0.8959 | 1.2482 | ✓ | ✗ | **rollback** |
| crf | 100 | 1.1400 | 1.0894 | +0.0506 | [-0.0201, +0.1213] | 0.5367 | 1.2495 | ✓ | ✗ | **rollback** |
| diversity | 100 | 1.0200 | 1.0894 | -0.0694 | [-0.2505, +0.1117] | 0.4531 | 1.2699 | ✓ | ✗ | **rollback** |

蓝球对照（基线 1/16 = 0.0625）：

| 特征 | 蓝球命中率 | Δ vs 0.0625 | 判定 |
|---|---:|---:|---|
| ac | 0.0900 | +0.0275 | rollback |
| entropy | 0.1100 | +0.0475 | rollback |
| hot_cold | 0.0900 | +0.0275 | rollback |
| crf | 0.0900 | +0.0275 | rollback |
| diversity | 0.0900 | +0.0275 | rollback |

## 基线策略对照（horizon=100，独立运行）

| 策略 | kind | 红球 mean_hits | 判定 | 说明 |
|---|---|---:|---|---|
| random | baseline | 1.1100 | rollback | 与理论期望 1.0909 一致（±0.3 内），MC 基线机制自洽 |
| freq | baseline | 1.1600 | rollback | 近 200 期频次 top6，未显著优于随机 |
| model:pg | model（冻结 PG 概率） | 1.0333（horizon=30） | rollback | PG 6 模型集成概率，冻结非滚动，结果仅参考 |

## 逐项结论与处置

1. **ac（AC 值分散度）→ rollback**：mean_hits 1.0800 < 基线 1.0894，Δ=-0.0094，p=0.9084，远不显著。AC 值作为选号偏好未带来命中增益。处置：`FEATURE_TOGGLES["ac"]` 维持 False（默认已关闭）。
2. **entropy（低熵可预测号）→ rollback**：Δ=+0.0206，p=0.8209，不显著；但为 5 特征中红球第二高（1.1100），蓝球命中率 0.1100 为最高。当前证据不足，**回退关闭**；若后续扩大 horizon 或换窗口可复核。
3. **hot_cold（冷热均值回归）→ rollback**：Δ=+0.0106，p=0.8959，不显著。处置：`FEATURE_TOGGLES["hot_cold"]` 回退为 False。
4. **crf（约束 beam 解码）→ rollback**：Δ=+0.0506，p=0.5367，红球 mean_hits 1.1400 为 5 特征最高，但未越过阈值 1.2495（差 0.1095）。**最接近显著的候选**，但按裁定规则仍回退；保留 ml/decode.py 供后续复用，开关默认 False。
5. **diversity（注单去重分散）→ rollback**：Δ=-0.0694，p=0.4531，5 特征中最差，且跨期状态在单注回测中无增益。处置：回退关闭。

## 结论

5 项特征在 horizon=100、train_min=800、MC N=1000 的标准化回测下**均未显著优于随机基线**（p 值全部 > 0.45，无一越过 基线+1.96SE 阈值）。按 Q3 裁定规则全部 **rollback**。

这是**符合预期的科学结论**（特征不显著 → 回退即验收通过），不是失败：
- 回测框架本身工作正常：random 策略自测命中 1.1100 ≈ 理论 1.0909，MC 基线 1.0894 ≈ 理论值，机制自洽；
- 5 项特征结果与 Dev 自报逐项一致（ac 1.08 / entropy 1.11 / hot_cold 1.10 / crf 1.14 / diversity 1.02），Dev 交付可信；
- 建议：生产 `FEATURE_TOGGLES` 仅保留 entropy/hot_cold 当前值并回退 hot_cold；ac/crf/diversity 保持关闭。特征管线代码（calc_ac_features / constrained_decode / keep_override）保留，供后续特征研究复用。

## 复现方式

```bash
cd /home/hermes/workspace/python/SSQ
.venv/bin/python evaluate.py --features ac,entropy,hot_cold,crf,diversity --horizon 100 --train-min 800 --out /home/hermes/workspace/data-center/ssq/analysis/results/feature_validation.json
.venv/bin/python evaluate.py --strategy random --horizon 100 --out /home/hermes/workspace/data-center/ssq/analysis/results/validation_baseline_random.json
.venv/bin/python evaluate.py --strategy freq --horizon 100 --out /home/hermes/workspace/data-center/ssq/analysis/results/validation_baseline_freq.json
```

同 seed（默认 1）同 horizon 两次运行结果逐字段一致（已验证可复现）。
