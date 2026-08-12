# SSQ 项目代码清理计划

## Context

用户要求做"减法"：去掉所有不必要的代码和输出信息，只保留以下 6 条命令所涉及的代码，并清理因测试/debug 引入的不规范代码：

```
python -m ml.main train-predict-batch --model rf --columns all
python -m ml.main train-predict-batch --model lgbm --columns all
python -m ml.main train-predict --model lstm_all
python -m ml.main train-predict --model lstm_blue
python -m ml.main train-predict --model lstm_reds
python -m ml.main train-predict --model cnn_math
```

**要删除的模型**：`cnn_1d`、`cnn_6_1`
**要删除的命令**：`train`、`predict`、`train-all`、`predict-all`、`status`、`interactive_mode`
**要删除的历史脚本**：`ml/train.py`、`ml/predict.py`（用户已确认删除）
**要删除的孤儿产物**：`ml/saved_models/cnn_6_1/`（用户已确认删除）

## 执行顺序（自底向上，避免行号漂移）

### 步骤 1：删除历史脚本和孤儿产物
- 删除 `ml/train.py`
- 删除 `ml/predict.py`
- 删除 `ml/saved_models/cnn_6_1/` 目录

### 步骤 2：清理 `ml/models/cnn_model.py`
**删除的类/函数**：
- `_TimeSeriesDataset`（L34-46，仅被 CNN1DModel 使用）
- `_CNN1DNet`（L64-87，1D-CNN 网络）
- `_HybridMultiTaskCNN6_1`（L91-115，6-1 CNN 网络）
- `CNN1DModel` 整个类（L196-438）
- `CNN6_1Model` 整个类（L442-816）

**保留**：`_HybridDataset`、`_HybridMathCNN`、`_compute_joint_loss`、`CNNMathModel`

**修改文件头 docstring**：`包含 1D-CNN、6-1 混合 CNN、数学增强 CNN 三种模型` → `包含数学增强 CNN 模型`

### 步骤 3：清理 `ml/models/__init__.py`
- import 语句删除 `CNN1DModel, CNN6_1Model`
- `__all__` 删除 `"CNN1DModel"` 和 `"CNN6_1Model"`

### 步骤 4：清理 `ml/config.py`
- `CNN_CONFIG` 删除 `"cnn1d"` 和 `"cnn6_1"` 子项（保留 `"cnn_math"`）
- `ModelType` 类删除 `CNN_1D = "cnn_1d"` 和 `CNN_6_1 = "cnn_6_1"`
- `ALL` 列表删除 `CNN_1D, CNN_6_1`
- `get_model_config` 删除 `CNN_1D` 和 `CNN_6_1` 的映射

### 步骤 5：清理 `ml/main.py`（从文件末尾向开头逐段处理）

#### 5.1 文件头 docstring（L1-17）
重写为仅保留 6 条命令的 Usage 示例。

#### 5.2 `_MODEL_CLASS_MAP`（L65-74）
删除 `CNN_1D` 和 `CNN_6_1` 的映射条目。

#### 5.3 `MODEL_INFO`（L81-130）
删除 `CNN_1D` 和 `CNN_6_1` 两个条目。

#### 5.4 删除死代码函数
- `get_model_save_dir`（L167-175）—— 仅被待删函数调用
- `model_exists`（L178-184）—— 仅被待删函数调用

#### 5.5 `train_individual`（L188-240）
- docstring 从 `（RF/LightGBM/CNN_1D）` 改为 `（RF/LightGBM）`

#### 5.6 `train_cnn`（L325-475）核心简化
- docstring 从 `训练 CNN 模型（1d/6_1/math）` 改为 `训练 CNN 数学增强模型`
- 删除 `column_name` 参数
- 删除 `if model_type == ModelType.CNN_1D:` 分支（L360-370）
- 删除 `elif model_type == ModelType.CNN_6_1:` 分支（L372-411，含 debug 日志）
- 将 `elif model_type == ModelType.CNN_MATH:` 改为直接逻辑（去 if 包裹）
- `run_train` 中调用处同步修改参数

#### 5.7 `predict_individual`（L524-643）
- docstring 从 `（RF/LightGBM/CNN_1D）` 改为 `（RF/LightGBM）`
- 删除 `if model_type == ModelType.CNN_1D:` 相关分支

#### 5.8 `predict_cnn`（L800-1037）核心简化
- 删除 `column_name` 参数
- 删除 `if model_type == ModelType.CNN_1D:` 分支（L827-863）
- 删除 `elif model_type == ModelType.CNN_6_1:` 分支（L865-946）
- 将 `elif model_type == ModelType.CNN_MATH:` 改为直接逻辑
- `run_predict` 中调用处同步修改参数

#### 5.9 `save_prediction`（L1040-1100）
- 注释从 `（RF/LGBM/CNN_1D）` 改为 `（RF/LGBM）`

#### 5.10 删除整段函数（从尾向头）
- `interactive_mode`（L1468-1651）
- `show_status`（L1404-1465）
- `batch_predict_all`（L1223-1273）
- `batch_train_all`（L1151-1221）
- 对应的分节注释

#### 5.11 `main()` 函数
- **argparse epilog** 重写为仅含保留命令
- **删除子 parser**：`train`、`predict`、`train-all`、`predict-all`、`status`
- **`train-predict` parser**：删除 `--column` 参数（RF/LGBM 走 batch 命令）
- **无参数行为**：从进入交互模式改为 `parser.print_help()`
- **删除命令分派**：`status`、`train`、`predict`、`train-all`、`predict-all` 分支
- **保留**：`train-predict` 和 `train-predict-batch` 分派分支

#### 5.12 清理重复 print（可选优化）
删除与 `logger.info` 完全同义的调试型 `print` 语句，保留面向用户的进度提示型 print（如 `print_banner`、`print("加载数据...")`）。

## 验证步骤

1. **语法检查**：`python -c "import ml.main; import ml.config; import ml.models"`
2. **枚举检查**：`python -c "from ml.config import ModelType; print(ModelType.ALL)"` 应输出 6 个模型
3. **残留引用扫描**：`rg -n "CNN_1D|CNN_6_1|CNN1DModel|CNN6_1Model" ml/ --type py` 应仅在 `ml/legacy/` 下命中
4. **逐条运行 6 条命令**验证功能正常
5. **无参数行为**：`python -m ml.main` 应打印帮助而非进入交互模式

## 风险与注意
- `ml/legacy/` 目录不动（历史探索代码，已隔离）
- 行号基于当前文件状态，实施时从文件末尾向开头逐段删除
- `--column` 移除后，误用 `train-predict --model rf` 会触发 argparse 报错（期望行为）
