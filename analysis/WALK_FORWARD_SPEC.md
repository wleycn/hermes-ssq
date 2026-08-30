# 滚动验证与对比测试设计（缺陷2/3/5 + 用户要求"对比测试"）

## 核心思想
时间序列不能用随机 train_test_split（会把未来泄露进训练）。改用 **walk-forward**：
用第 1..k 期训练 → 预测第 k+1 期 → 滑窗推进 → 汇总所有真实外推期的命中率。

## 实现：`ml/eval/walk_forward.py`

### 函数签名
```
def walk_forward(
    model_factory,          # 给定训练数据返回训练好的模型
    df,                     # 全量特征 DataFrame（已统一特征）
    train_min=500,          # 最少训练期数
    step=1,                 # 每步外推 1 期
    max_horizon=200,        # 最多外推多少期（控制总时长）
    target="red_set",       # "red_set" | "blue" | "all"
) -> dict:
```

### 流程
1. 取 `df[train_min : train_min+max_horizon]`，对每一期 `t`：
   - 训练集 = `df[0 : train_min + i]`（i 为已外推步数）
   - 用训练集训练模型（轻量，控制轮数避免太慢）
   - 预测第 t 期
   - 记录集合命中 / 蓝球命中
2. 汇总：平均集合命中、命中分布、蓝球 top1 命中率。

### 基线对照（必须同口径跑）
- `random_baseline`：每期随机选 6 号，重复多次取平均重叠（理论 1.09）。
- `freq_baseline`：用训练期最频繁 6 号（实证已得 ≈1.11）。
- 模型命中率应与之并列。

## 对比测试矩阵（最终报告的一张表）
| 模型 | 建模方式 | 平均集合命中 | vs 随机基线 | 备注 |
|---|---|---|---|---|
| RF | 旧(分位置) | ? | ? | 缺陷3 指标失真 |
| RF | 新(集合) | ? | ? | |
| LGBM | 旧 | ? | ? | |
| LGBM | 新 | ? | ? | |
| LSTM_REDS | 旧 | ? | ? | |
| SetRed(LSTM) | 新 | ? | ? | |
| CNN_MATH | 旧(后处理修正后) | ? | ? | 缺陷4 泄漏已修 |
| 随机基线 | — | 1.09 | 基准 | |
| 频率基线 | — | 1.11 | +0.02 | |

预期结论：所有模型滚动命中率 ≈ 随机基线，证明开奖不可预测；旧模型"指标漂亮"只是评估失真。

## 性能约束
- 3487 期，walk-forward 跑 200 期、每期训练：RF/LGBM 快；LSTM/CNN 每期重训慢。
- 折中：LSTM/CNN 用**增量窗口 + 限制 epochs**（如 30），或只跑较小 horizon（如 100 期）以控制总时长。
- 先验证小样本跑通，再决定全量。

## 验证
- 跑通后输出 CSV：`/home/hermes/workspace/data-center/ssq/analysis/results/walk_forward_<model>_<mode>.csv`
- 汇总表存 `/home/hermes/workspace/data-center/ssq/analysis/results/comparison_table.csv`
