# 特征现状审计 (T1)

> 审计时间: 2026-08-13
> 审计范围: `ml/features/feature_engineer.py` 特征工程与选号链路
> 状态: 定稿 (5 项特征)

## 审计结论

| # | 特征 | 状态 | 代码位置 | 说明 |
|---|------|------|----------|------|
| 1 | AC值 (AC Value) | ❌ 未实现 | — | 无 AC 值特征计算/列；未列入 `UNIFIED_KEEP` 白名单 |
| 2 | 窗口熵 (Window Entropy) | ✅ 已实现 | `ml/features/feature_engineer.py` `calc_entropy_features()` (L363)；产出 `Entropy` 列 (L387)；白名单 `UNIFIED_KEEP` (L939-959, `Entropy` 在 L944) | 滑动窗口信息熵，已进入统一特征空间 |
| 3 | 冷热指数 (Hot/Cold) | ✅ 已实现 | `ml/features/feature_engineer.py` `calc_hot_cold_features()` (L418)；产出 `Hot_Count`/`Cold_Count`/`Hot_Cold_Ratio` (L459-461)；白名单 `UNIFIED_KEEP` (L939-959, 在 L948) | 另含 `Hot_Count_Mean`/`Cold_Count_Mean` (L467-470) |
| 4 | CRF 约束解码 | ❌ 未实现 | — | 无 CRF / 约束解码模块 |
| 5 | 多样性采样 | ❌ 未实现 | — | 无去重/多样性指标采样；本期 P1 冷门组合加权 (`ml/popularity.py`) 为相关第一步，但多样性采样本身仍缺失 |

## 备注

- 已实现项均通过 `build_unified_features()` 的 `UNIFIED_KEEP` 白名单 (L939-959) 进入模型特征空间。
- 未实现项为后续迭代候选；本期 (P0 旋转矩阵 + P1 冷门组合加权) 不扩展特征工程（Dev-2 范围）。
