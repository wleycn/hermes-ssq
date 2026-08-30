# CNN_MATH 后处理去泄漏（缺陷4 修复设计）

## 问题描述
`ml/models/cnn_model.py::CNNMathModel.predict_with_post_processing` 存在两类问题：
1. **信息泄漏**：`Norm_Mean / Norm_Std`（红球和值的移动统计）在预测时取 `df.iloc[-1]`，即**最新一期**统计，再用它"校正"对下期的预测和值 → 用了未来信息。
2. **魔法数字无依据**：entropy_chaos_threshold=4.0、sum_constraint_threshold=10、chaos_damping_factor=0.5 等是拍脑袋值，无敏感性分析。

## 修复方案

### 1. 去泄漏
- 和值约束用的均值/标准差，改为**训练期滚动统计**：在 walk-forward 中，用 `df[0:train_end]` 计算，绝不用 `df[-1]`。
- 把"约束所用的统计量"作为参数传入，而非从全局 df 末行偷取。

### 2. 魔法数字可配置 + 敏感性标注
- 全部阈值移入 config（已在 `CNN_CONFIG["cnn_math"]`）。
- 在对比报告中加一节：把 threshold 在 {3,4,5,6} 与 damping {0.3,0.5,0.7} 网格扫一遍，看最终命中率是否敏感 → 若命中率对阈值不敏感，说明后处理是摆设，进一步佐证"无信号"。

### 3. 泊松/熵后处理同样改为训练期
- `Poisson_R{i}` 当前取 `df.iloc[-1]` → 改为传入"截至训练末期的泊松概率"。

## 验证
- 单元测试：构造最小 df，断言 `predict_with_post_processing` 不读取任何 `iloc[-1]` 之后的信息（难以直接断言，改为 code review + 输入仅含训练期数据）。
- 敏感性扫描输出：`/home/hermes/workspace/data-center/ssq/analysis/results/cnn_threshold_sweep.csv`。
