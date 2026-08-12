# 统一特征入口规约（缺陷6 修复设计）

## 现状问题
- `ml/models/lgb_model.py::calculate_all_features` 仅 8 类特征（约 40 列），RF/LGBM 用此入口。
- `ml/features/feature_engineer.py::FeatureEngineer.compute_all_features` 约 20 类（上百列），LSTM/CNN 用此入口。
- 同一号码在不同模型"看到"的特征空间不一致 → 模型间不可比，结论失真。

## 目标
单一特征事实源。所有模型训练/预测都走同一套特征 + 同一套筛选。

## 实现方案
1. 在 `ml/features/feature_engineer.py` 新增公共函数：
   `build_unified_features(df, mode="full"|"compact") -> pd.DataFrame`
   - 内部调用 `compute_all_features` 生成全部候选特征。
   - `mode="compact"` 落到一个**去冗余白名单**，去掉强共线派生列。

2. 白名单（compact 模式保留，约 30~45 列，按语义独立分组）：
   - 基础: Sum, OddRatio, BigRatio
   - 奇偶/大小逐位: Odd_Even_1..6（仅 LSTM 序列用，可不用）
   - 频率: Freq_1..33（每个红球历史累计出现）
   - 近期频率: Recent_Freq_1..33（滑动窗口）
   - 遗漏: Last_Appear_1..33（距上次出现期数）
   - 冷热: Hot_Count, Cold_Count, Hot_Cold_Ratio, Hot_Count_Mean, Cold_Count_Mean
   - 区间: Int_1_11, Int_12_22, Int_23_33, Int_Dist_Max, Int_Dist_Min, Int_Dist_Std
   - 和值统计: Sum_Mean, Sum_Std, Sum_Skew, Sum_Kurt
   - 和值区间: Sum_Int_1..6 + 其滚动频率
   - 质数: Prime_Count, Prime_Ratio, Prime_Count_Mean
   - 连号: Consecutive_Pairs, Max_Consecutive
   - 熵: Entropy
   - 马尔可夫: Markov_Odd_Prob_0..6, Markov_Big_Prob_0..6
   - 位置趋势: Pos_i_Mean, Pos_i_Std（i=1..6）
   - 大数定律偏差: LLN_Abs_Deviation_Mean

3. **明确剔除**（避免共线/泄漏）：
   - 各种 `*_Ratio` 与其分子列同时存在的只留其一。
   - `calc_onehot_features` 的 `Last_Draw_i`（用 shift(1) 易串期，且和 Recent_Freq 信息重叠）→ 默认不进 compact。
   - `LLN_Max/Min_Deviation`（与 Abs_Mean 共线）→ 只留 Abs_Mean。
   - `Unique_Count`（与 Freq_* 信息重叠）→ 剔除。

4. 模型接入：
   - `rf_model.prepare_data` / `lgb_model.prepare_data`：改用 `build_unified_features(df, "compact")` 后取所需列。
   - `lstm_model` / `cnn_model` 的 `prepare_data`：feature_cols 直接来自同一函数的列名列表。
   - 保证 `extract_feature_columns` 与白名单一致。

## 验证
- 装好依赖后跑：`python -c "from ml.features.feature_engineer import build_unified_features; ..."` 确认列数稳定、无 NaN 爆炸。
- 各模型 prepare_data 输出特征维度打印一致。
