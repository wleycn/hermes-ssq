import pandas as pd
import numpy as np
import lightgbm as lgb
import joblib
from scipy.stats import skew, kurtosis
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.metrics import accuracy_score, log_loss
from sklearn.preprocessing import LabelEncoder
from pathlib import Path
import time
import os
import json
from datetime import datetime

'''
Light Gradient Boosting Machine: 轻量级梯度提升机
'''

# ================= 全局参数配置 =================
# ========== 数据相关参数 ==========
WINDOW_SIZE = 128           # 滑动窗口大小：用于构建特征的历史期数，值越大，模型能学习更长期的历史模式，但计算成本增加，推荐范围：64-256，根据数据量和计算资源调整
WINDOW_STEP = 1             # 滑动步长：窗口每次滑动的期数
TOP_PRO = 6                 # 预测前N个号码：输出概率最高的N个候选号码
TARGET_COLS = ['Red1', 'Red2', 'Red3', 'Red4', 'Red5', 'Red6', 'Blue1']

# ========== 模型训练参数 ==========
BOOST_ROUND = 500           # 最大迭代轮数：LightGBM的boosting迭代次数上限，迭代次数越多，模型越复杂，可能过拟合，通常配合早停机制，实际迭代次数会小于此值
STOP_ROUND = 13             # 早停轮数：验证损失连续N轮不下降则停止训练，用于防止过拟合，避免无效迭代，推荐范围：10-50，值越大训练时间越长
RETRAIN_MODEL = 'Y'         # 是否重新训练模型：'Y'重新训练，'N'加载已有模型，首次运行或修改参数后需设为'Y'，后续预测时可设为'N'以节省时间

# ========== 模型调参网格 ==========
param_grid = {
    'num_leaves': [3,22,25,40,42],       # 叶子节点数：控制模型复杂度的核心参数，值越大，模型越复杂，容易过拟合，推荐范围：3-31，根据数据量和特征数调整，本脚本约94个特征，建议num_leaves ≤ 特征数
    'learning_rate': [0.003],       # 学习率：每步迭代对模型权重的更新幅度，值越大，收敛越快但可能错过最优解，值越小，收敛越慢但更稳定，推荐范围：0.001-0.1，配合boost_round调整
    'reg_alpha': [0],               # L1正则化系数：对权重绝对值的惩罚，用于特征选择，防止过拟合，值越大，模型越稀疏（更多权重为0）
    'reg_lambda': [0.05]          # L2正则化系数：对权重平方的惩罚，用于防止过拟合，使权重分布更平滑，推荐范围：0-0.1，与reg_alpha配合使用
}

# ========== 模型保存目录 ==========
MODEL_DIR = Path(r"./models")
MODEL_DIR.mkdir(parents=True, exist_ok=True)

# ========== 全局变量 ==========
predictions_list = []
data_list = []
model_metrics = {}
column_times = {}


# ================= 工具函数 =================
def sliding_window_numpy(data_list, window_size, step):
    """使用 NumPy 内存视图实现高性能滑动窗口拆解"""
    arr = np.asarray(data_list)
    n = arr.shape[0]
    num_windows = (n - window_size) // step + 1
    strides = arr.strides[0]
    new_shape = (num_windows, window_size)
    new_strides = (strides * step, strides)
    windows = np.lib.stride_tricks.as_strided(arr, shape=new_shape, strides=new_strides)
    return pd.DataFrame(windows)


def top_k_accuracy(y_true, y_pred_proba, k=TOP_PRO, **kwargs):
    """计算 Top-K 命中率（向量化高性能版）"""
    if y_pred_proba.ndim < 2 or y_pred_proba.shape[1] < 2:
        return 0.0
    top_k = min(k, y_pred_proba.shape[1])
    top_indices = np.argsort(y_pred_proba, axis=1)[:, -top_k:]
    correct = np.any(top_indices == y_true.reshape(-1, 1), axis=1)
    return np.mean(correct)


def analyze_overfitting(train_loss, test_loss, train_acc, test_acc):
    """分析模型是否过拟合/欠拟合（分级判断）"""
    loss_gap = test_loss - train_loss
    acc_gap = train_acc - test_acc

    if train_loss < 1.0 and loss_gap > 1.5:
        return {
            'status': '过拟合',
            'severity': '严重',
            'reason': f'训练损失({train_loss:.4f})极低，测试损失({test_loss:.4f})远高于训练损失, 差距={loss_gap:.4f}',
            'suggestion': '减小num_leaves, 增加reg_alpha/reg_lambda, 降低learning_rate'
        }
    if loss_gap > 0.8 or acc_gap > 0.3:
        return {
            'status': '过拟合',
            'severity': '中度',
            'reason': f'训练命中率({train_acc:.4f})高于测试命中率({test_acc:.4f}), 差距={acc_gap:.4f}; 损失差距={loss_gap:.4f}',
            'suggestion': '减小num_leaves, 增加正则化强度, 降低learning_rate'
        }
    if acc_gap > 0.15:
        return {
            'status': '轻微过拟合',
            'severity': '轻度',
            'reason': f'训练命中率({train_acc:.4f})高于测试命中率({test_acc:.4f}), 差距={acc_gap:.4f}',
            'suggestion': '适当减小模型复杂度，或增加训练数据'
        }
    if train_loss > 3.0 and test_loss > 3.0:
        return {
            'status': '欠拟合',
            'severity': '严重' if train_loss > 4.0 else '中度',
            'reason': f'训练损失({train_loss:.4f})和测试损失({test_loss:.4f})都很高',
            'suggestion': '增加num_leaves, 增加boost_round, 降低正则化强度'
        }
    return {
        'status': '训练良好',
        'severity': '正常',
        'reason': f'训练损失({train_loss:.4f})与测试损失({test_loss:.4f})接近, 命中率差距={acc_gap:.4f}',
        'suggestion': '当前参数合适，可尝试微调'
    }


# ========== 特征工程常量 ==========
PRIMES = {2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31}  # 1-33范围内的质数集合
                                                       # 用于计算质数特征：质数数量、质数比例

def calc_statistical_features(window_data):
    """统计特征：和值、均值、方差、偏度、峰度"""
    features = {}
    row_values = window_data.values
    
    features['Sum'] = row_values.sum(axis=1)
    features['Mean'] = row_values.mean(axis=1)
    features['Std'] = row_values.std(axis=1)
    features['Min'] = row_values.min(axis=1)
    features['Max'] = row_values.max(axis=1)
    features['Range'] = features['Max'] - features['Min']
    
    features['Skew'] = skew(row_values, axis=1)
    features['Kurtosis'] = kurtosis(row_values, axis=1)
    
    return pd.DataFrame(features)

def calc_frequency_features(window_data):
    """频率特征：每个数字在窗口中的出现次数"""
    features = {}
    row_values = window_data.values
    
    for num in range(1, 34):
        features[f'Freq_{num}'] = (row_values == num).sum(axis=1)
    
    features['Unique_Count'] = pd.DataFrame(row_values).nunique(axis=1).values
    
    return pd.DataFrame(features)

def calc_interval_features(window_data):
    """区间特征：各区间数字数量"""
    features = {}
    row_values = window_data.values
    
    features['Int_1_11'] = ((row_values >= 1) & (row_values <= 11)).sum(axis=1)
    features['Int_12_22'] = ((row_values >= 12) & (row_values <= 22)).sum(axis=1)
    features['Int_23_33'] = ((row_values >= 23) & (row_values <= 33)).sum(axis=1)
    
    features['Int_Max'] = np.max([features['Int_1_11'], features['Int_12_22'], features['Int_23_33']], axis=0)
    features['Int_Min'] = np.min([features['Int_1_11'], features['Int_12_22'], features['Int_23_33']], axis=0)
    
    return pd.DataFrame(features)

def calc_odd_even_features(window_data):
    """奇偶特征：奇数数量、偶数数量、奇偶比"""
    features = {}
    row_values = window_data.values
    
    features['Odd_Count'] = (row_values % 2 == 1).sum(axis=1)
    features['Even_Count'] = (row_values % 2 == 0).sum(axis=1)
    features['Odd_Even_Ratio'] = features['Odd_Count'] / (features['Even_Count'] + 1e-6)
    
    return pd.DataFrame(features)

def calc_size_features(window_data):
    """大小特征：大号数量、小号数量、大小比（以17为界）"""
    features = {}
    row_values = window_data.values
    
    features['Big_Count'] = (row_values > 16).sum(axis=1)
    features['Small_Count'] = (row_values <= 16).sum(axis=1)
    features['Big_Small_Ratio'] = features['Big_Count'] / (features['Small_Count'] + 1e-6)
    
    return pd.DataFrame(features)

def calc_consecutive_features(window_data):
    """连号特征：连号对数、最长连号"""
    features = {'Consecutive_Pairs': [], 'Max_Consecutive': []}
    row_values = window_data.values
    
    for row in row_values:
        sorted_row = np.sort(row)
        diffs = np.diff(sorted_row)
        consecutive_pairs = np.sum(diffs == 1)
        
        max_consecutive = 1
        current = 1
        for d in diffs:
            if d == 1:
                current += 1
                max_consecutive = max(max_consecutive, current)
            else:
                current = 1
        
        features['Consecutive_Pairs'].append(consecutive_pairs)
        features['Max_Consecutive'].append(max_consecutive)
    
    features['Consecutive_Ratio'] = np.array(features['Consecutive_Pairs']) / (len(window_data.columns) - 1)
    
    return pd.DataFrame(features)

def calc_prime_features(window_data):
    """质数特征：质数数量、质数比"""
    features = {}
    row_values = window_data.values
    
    features['Prime_Count'] = np.isin(row_values, list(PRIMES)).sum(axis=1)
    features['Prime_Ratio'] = features['Prime_Count'] / len(window_data.columns)
    
    return pd.DataFrame(features)

def calc_position_features(window_data):
    """位置特征：最大值位置、最小值位置、中位数位置"""
    features = {}
    row_values = window_data.values
    
    features['Max_Position'] = np.argmax(row_values, axis=1)
    features['Min_Position'] = np.argmin(row_values, axis=1)
    
    median_positions = []
    for row in row_values:
        sorted_indices = np.argsort(row)
        median_positions.append(sorted_indices[len(row) // 2])
    features['Median_Position'] = np.array(median_positions)
    
    return pd.DataFrame(features)

def calculate_all_features(window_data):
    """计算所有特征"""
    all_features = [window_data]
    
    feature_funcs = [
        ('统计特征', calc_statistical_features),
        ('频率特征', calc_frequency_features),
        ('区间特征', calc_interval_features),
        ('奇偶特征', calc_odd_even_features),
        ('大小特征', calc_size_features),
        ('连号特征', calc_consecutive_features),
        ('质数特征', calc_prime_features),
        ('位置特征', calc_position_features),
    ]
    
    for name, func in feature_funcs:
        try:
            feat_df = func(window_data)
            all_features.append(feat_df)
        except Exception as e:
            print(f"  ⚠️ {name}计算失败: {e}")
    
    return pd.concat(all_features, axis=1)

# ================= 核心处理流程 =================
def prepare_data(column_name, df):
    """子流程 1：数据准备与编码"""
    t_start = time.time()

    r1 = sliding_window_numpy(df[column_name], window_size=WINDOW_SIZE, step=WINDOW_STEP)
    lastColName = r1.columns[-1]
    r1 = r1.rename(columns={lastColName: 'label'})

    X = r1.iloc[:, :-1]
    y_raw = r1.iloc[:, -1]

    value_counts = y_raw.value_counts()
    valid_labels = value_counts[value_counts >= 6].index
    valid_mask = y_raw.isin(valid_labels)

    X = X[valid_mask].reset_index(drop=True)
    y_raw = y_raw[valid_mask].reset_index(drop=True)

    X = calculate_all_features(X)

    le = LabelEncoder()
    y_encoded = le.fit_transform(y_raw)
    num_classes = len(le.classes_)

    elapsed = time.time() - t_start

    print(f"  ✓ 数据准备完成")
    print(f"    特征维度: {X.shape}, 类别数: {num_classes}, 有效样本: {len(X)}")
    print(f"    耗时: {elapsed:.2f}s")

    return X, y_encoded, le, num_classes


def train_base_model(X, y_encoded, num_classes, column_name):
    """子流程 2：基础 LightGBM 模型训练与评估"""
    t_start = time.time()

    X_train, X_test, y_train, y_test = train_test_split(X, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded)
    train_data = lgb.Dataset(X_train, label=y_train)
    test_data = lgb.Dataset(X_test, label=y_test, reference=train_data)

    params = {
        'objective': 'multiclass',
        'num_class': num_classes,
        'metric': 'multi_logloss',
        'boosting_type': 'gbdt',
        'verbose': -1,
        'seed': 42
    }

    callbacks = [
        lgb.early_stopping(stopping_rounds=STOP_ROUND, verbose=False),
        lgb.log_evaluation(period=-1)
    ]

    model = lgb.train(
        params,
        train_data,
        num_boost_round=BOOST_ROUND,
        valid_sets=[test_data],
        callbacks=callbacks
    )

    y_pred_proba_train = model.predict(X_train)
    y_pred_proba_test = model.predict(X_test)
    y_pred_train = np.argmax(y_pred_proba_train, axis=1)
    y_pred_test = np.argmax(y_pred_proba_test, axis=1)

    all_classes = np.arange(num_classes)
    train_log_loss = log_loss(y_train, y_pred_proba_train, labels=all_classes)
    test_log_loss = log_loss(y_test, y_pred_proba_test, labels=all_classes)

    train_top_k_acc = top_k_accuracy(y_train, y_pred_proba_train)
    test_top_k_acc = top_k_accuracy(y_test, y_pred_proba_test)

    train_accuracy = accuracy_score(y_train, y_pred_train)
    test_accuracy = accuracy_score(y_test, y_pred_test)

    val_loss = model.best_score['valid_0']['multi_logloss']

    overfit_analysis = analyze_overfitting(train_log_loss, test_log_loss, train_top_k_acc, test_top_k_acc)

    elapsed = time.time() - t_start

    top_k_label = f'Top-{TOP_PRO}命中率'

    print(f"\n  {'='*50}")
    print(f"  📊 基础模型评估报告")
    print(f"  {'='*50}")
    print(f"  {'指标':<20} {'训练集':<15} {'测试集':<15}")
    print(f"  {'-'*50}")
    print(f"  {'对数损失(Log Loss)':<20} {train_log_loss:<15.6f} {test_log_loss:<15.6f}")
    print(f"  {top_k_label:<20} {train_top_k_acc:<15.4f} {test_top_k_acc:<15.4f}")
    print(f"  {'准确率(Accuracy)':<20} {train_accuracy:<15.4f} {test_accuracy:<15.4f}")
    print(f"  {'验证损失(Best)':<20} -              {val_loss:<15.6f}")
    print(f"  {'-'*50}")
    print(f"  📈 拟合状态: {overfit_analysis['status']} ({overfit_analysis['severity']})")
    print(f"     原因: {overfit_analysis['reason']}")
    print(f"     建议: {overfit_analysis['suggestion']}")
    print(f"  {'='*50}")
    print(f"  耗时: {elapsed:.2f}s")

    metrics = {
        'train_log_loss': train_log_loss,
        'test_log_loss': test_log_loss,
        'train_top_k_acc': train_top_k_acc,
        'test_top_k_acc': test_top_k_acc,
        'train_accuracy': train_accuracy,
        'test_accuracy': test_accuracy,
        'val_loss': val_loss,
        'overfit_status': overfit_analysis['status'],
        'sample_count': {'train': len(X_train), 'test': len(X_test)}
    }
    model_metrics[column_name + '_base'] = metrics

    return model, X_train, y_train, X_test, y_test


def tune_model(X_train, y_train, num_classes, column_name):
    """子流程 3：手动网格搜索调参（避免 make_scorer 兼容性问题）"""
    t_start = time.time()

    kfold = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)
    all_params = []

    for num_leaves in param_grid['num_leaves']:
        for lr in param_grid['learning_rate']:
            for alpha in param_grid['reg_alpha']:
                for lam in param_grid['reg_lambda']:
                    all_params.append({
                        'num_leaves': num_leaves,
                        'learning_rate': lr,
                        'reg_alpha': alpha,
                        'reg_lambda': lam
                    })

    best_score = -1.0
    best_params = None
    all_cv_scores = []
    all_train_scores = []

    print(f"  正在网格搜索调参 (共 {len(all_params)} 组参数)...")

    for params in all_params:
        cv_scores = []
        train_scores = []

        for train_idx, val_idx in kfold.split(X_train, y_train):
            X_tr = X_train.iloc[train_idx] if isinstance(X_train, pd.DataFrame) else X_train[train_idx]
            y_tr = y_train[train_idx]
            X_val = X_train.iloc[val_idx] if isinstance(X_train, pd.DataFrame) else X_train[val_idx]
            y_val = y_train[val_idx]

            train_data = lgb.Dataset(X_tr, label=y_tr)
            val_data = lgb.Dataset(X_val, label=y_val, reference=train_data)

            model_params = {
                'objective': 'multiclass',
                'num_class': num_classes,
                'metric': 'multi_logloss',
                'boosting_type': 'gbdt',
                'verbose': -1,
                'seed': 42,
                **params
            }

            callbacks = [
                lgb.early_stopping(stopping_rounds=STOP_ROUND, verbose=False),
                lgb.log_evaluation(period=-1)
            ]

            model = lgb.train(
                model_params,
                train_data,
                num_boost_round=BOOST_ROUND,
                valid_sets=[val_data],
                callbacks=callbacks
            )

            y_pred_proba_val = model.predict(X_val)
            y_pred_proba_tr = model.predict(X_tr)

            val_score = top_k_accuracy(y_val, y_pred_proba_val)
            tr_score = top_k_accuracy(y_tr, y_pred_proba_tr)

            cv_scores.append(val_score)
            train_scores.append(tr_score)

        mean_cv = np.mean(cv_scores)
        mean_train = np.mean(train_scores)
        all_cv_scores.append(mean_cv)
        all_train_scores.append(mean_train)

        if mean_cv > best_score:
            best_score = mean_cv
            best_params = params.copy()

    train_data_final = lgb.Dataset(X_train, label=y_train)

    best_model_params = {
        'objective': 'multiclass',
        'num_class': num_classes,
        'metric': 'multi_logloss',
        'boosting_type': 'gbdt',
        'verbose': -1,
        'seed': 42,
        **best_params
    }

    best_model = lgb.train(
        best_model_params,
        train_data_final,
        num_boost_round=BOOST_ROUND
    )

    y_pred_proba_train = best_model.predict(X_train)
    train_log_loss = log_loss(y_train, y_pred_proba_train, labels=np.arange(num_classes))

    importances = best_model.feature_importance()
    feature_names = [f'窗口位置_{i}' for i in range(len(importances))]
    top_features = sorted(zip(feature_names, importances), key=lambda x: -x[1])[:10]

    elapsed = time.time() - t_start

    print(f"\n  {'='*50}")
    print(f"  🎯 网格搜索调参结果")
    print(f"  {'='*50}")
    print(f"  {column_name} 最佳参数: {best_params}")
    print(f"  {'-'*50}")
    print(f"  {'指标':<20} {'值':<20}")
    print(f"  {'-'*50}")
    print(f"  {'交叉验证Top-K均值':<20} {best_score:<20.6f}")
    print(f"  {'交叉验证标准差':<20} {np.std(all_cv_scores):<20.6f}")
    print(f"  {'训练集Top-K均值':<20} {np.mean(all_train_scores):<20.6f}")
    print(f"  {'训练集对数损失':<20} {train_log_loss:<20.6f}")
    print(f"  {'-'*50}")
    # print(f"  🌟 特征重要性 Top-10:")
    # for i, (name, imp) in enumerate(top_features, 1):
    #     print(f"     {i:>2}. {name:<15} 重要性: {imp:.6f}")
    # print(f"  {'='*50}")
    print(f"  耗时: {elapsed:.2f}s")

    metrics = {
        'best_params': best_params,
        'cv_mean_score': best_score,
        'cv_std': np.std(all_cv_scores),
        'train_score_mean': np.mean(all_train_scores),
        'train_log_loss': train_log_loss,
        'top_features': top_features
    }
    model_metrics[column_name + '_tuned'] = metrics

    return best_model


def predict_and_evaluate(best_model, le, X_train_columns, df, column_name):
    """子流程 4：新数据预测与结果解码"""
    t_start = time.time()

    new_data = df[column_name].tail(WINDOW_SIZE - 1).tolist()
    original_columns = list(range(WINDOW_SIZE - 1))
    new_data_df = pd.DataFrame([new_data], columns=original_columns)
    new_data_df = calculate_all_features(new_data_df)

    y_pred_proba = best_model.predict(new_data_df)
    top_k_indices = np.argsort(y_pred_proba[0])[-TOP_PRO:]

    top_k_original_numbers = le.inverse_transform(top_k_indices)
    top_k_probs = y_pred_proba[0][top_k_indices]

    prediction_json = {
        "predict_time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "target_column": column_name,
        "top_k_predictions": [
            {"number": int(num), "probability": round(float(prob), 4)}
            for num, prob in zip(top_k_original_numbers, top_k_probs)
        ]
    }

    data_json = [
        {str(int(num)): round(float(prob), 4)}
        for num, prob in zip(top_k_original_numbers, top_k_probs)
    ]

    if column_name != "Blue1":
        predictions_list.append(prediction_json)
        data_list.extend(data_json)

    elapsed = time.time() - t_start

    print(f"\n  🎯 {column_name} 预测结果（按概率降序）:")
    print(f"  {'-'*35}")
    print(f"  {'MLType':<6} {'BallType':<6} {'数字':<8} {'概率':<10}")
    print(f"  {'-'*35}")
    for idx, (number, prob) in enumerate(zip(top_k_original_numbers, top_k_probs), 1):
        print(f"   {'LGBM':<6} {column_name:<6} {number:<8} {prob:<10.4f}")
    print(f"  {'-'*35}")
    print(f"  耗时: {elapsed:.2f}s")


def process_column(column_name, df):
    """主流程：根据全局常量 RETRAIN_MODEL 决定是训练还是直接预测"""
    t_start = time.time()

    print(f"\n{'='*60}")
    print(f" 🚀 开始处理: {column_name}")
    print(f"{'='*60}")

    model_path = MODEL_DIR / f"{column_name}_lgb_model.joblib"
    encoder_path = MODEL_DIR / f"{column_name}_label_encoder.joblib"
    columns_path = MODEL_DIR / f"{column_name}_columns.joblib"

    X, y_encoded, le, num_classes = prepare_data(column_name, df)

    if RETRAIN_MODEL.upper() == 'Y':
        print(f"  🔄 全局开关为 Y，开始重新训练模型...")

        _, X_train, y_train, _, _ = train_base_model(X, y_encoded, num_classes, column_name)

        best_model = tune_model(X_train, y_train, num_classes, column_name)

        joblib.dump(best_model, model_path)
        joblib.dump(le, encoder_path)
        joblib.dump(list(X_train.columns), columns_path)
        print(f"\n  ✓ 模型及编码器已保存至: {MODEL_DIR}")

        X_train_columns = list(X_train.columns)

    else:
        print(f"  📥 全局开关为 N，跳过训练，直接加载已有模型...")

        if not os.path.exists(model_path):
            raise FileNotFoundError(f"未找到已保存的模型文件: {model_path}。请先将 RETRAIN_MODEL 设置为 'Y' 进行训练！")

        best_model = joblib.load(model_path)
        le = joblib.load(encoder_path)
        X_train_columns = joblib.load(columns_path)

        print(f"  ✓ 模型加载成功！")

    predict_and_evaluate(best_model, le, X_train_columns, df, column_name)

    elapsed_time = time.time() - t_start
    column_times[column_name] = elapsed_time

    print(f"\n{'='*60}")
    print(f" ✅ 结束处理: {column_name}, 总耗时: {elapsed_time:.2f}秒")
    print(f"{'='*60}")


def print_global_report():
    """打印全局评估报告"""
    print(f"\n{'='*60}")
    print(f" 📊 全局评估报告")
    print(f"{'='*60}")

    print(f"\n--- 各列处理耗时统计 ---")
    print(f"{'列名':<10} {'耗时(秒)':<12}")
    print(f"{'---':<10} {'---':<12}")
    for col, t in column_times.items():
        print(f"{col:<10} {t:<12.2f}")

    print(f"\n--- 模型可信度分析 ---")
    print(f"{'列名':<10} {'拟合状态':<12} {'基础Top-K':<12} {'调优Top-K':<12}")
    print(f"{'---':<10} {'---':<12} {'---':<12} {'---':<12}")
    for col in TARGET_COLS:
        base_key = col + '_base'
        tuned_key = col + '_tuned'
        if base_key in model_metrics:
            base = model_metrics[base_key]
            tuned = model_metrics.get(tuned_key, {})
            status = base.get('overfit_status', '未知')
            base_acc = base.get('test_top_k_acc', 0)
            tuned_acc = tuned.get('cv_mean_score', '-')
            print(f"{col:<10} {status:<12} {base_acc:<12.4f} {str(tuned_acc)[:12]:<12}")

    total_time = sum(column_times.values())
    print(f"\n总耗时: {total_time:.2f}秒")
    print(f"{'='*60}")


# ================= 主程序入口 =================
if __name__ == "__main__":
    data_file = Path("C:/Users/lw25622/ML/SSQ/1.csv")

    print(f"{'='*60}")
    print(f" 🌲 LightGBM 双色球预测系统 - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"{'='*60}")

    print(f"\n📥 正在加载数据: {data_file}")
    df = pd.read_csv(data_file)
    print(f"✓ 数据加载成功, 共 {len(df)} 条记录, {len(df.columns)} 列")
    print(f"  耗时: 0.01秒")

    print(f"\n⚙️ 配置参数:")
    print(f"  窗口大小: {WINDOW_SIZE}, 步长: {WINDOW_STEP}, Top-K: {TOP_PRO}")
    print(f"  目标列: {TARGET_COLS}")
    print(f"  调参网格: {param_grid}")

    for col in TARGET_COLS:
        if col in df.columns:
            process_column(col, df)
        else:
            print(f"⚠️ 数据集中未找到列: {col}，已跳过。")

    sorted_data2 = sorted(
        data_list,
        key=lambda x: int(list(x.keys())[0]),
        reverse=False
    )

    file_path = datetime.now().strftime("%Y%m%d")
    file_path = "./" + file_path + ".csv"

    df = pd.DataFrame.from_records(sorted_data2, index=None).T.stack().reset_index()
    df = df.drop(columns=['level_1'])
    df.columns = ['number', 'probability']

    df.to_csv(file_path, index=False, mode='a')
    print(f"\n📄 数据已成功写入: {file_path}")

    print_global_report()