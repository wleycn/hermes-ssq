import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, KFold
from sklearn.metrics import accuracy_score, log_loss
from sklearn.preprocessing import LabelEncoder
from pathlib import Path
import time
from datetime import datetime

# ================= 全局参数配置 =================
# ========== 数据相关参数 ==========
WINDOW_SIZE = 165          # 滑动窗口大小
WINDOW_STEP = 1            # 滑动步长
TOP_PRO = 6                # 预测前N个号码
TARGET_COLS = ['Red1', 'Red2', 'Red3', 'Red4', 'Red5', 'Red6', 'Blue1']

# ========== 模型调参网格 ==========
param_grid = {
    'n_estimators': [64,75,150],
    'max_depth': [4,7,12],          # 更浅的树
    'min_samples_split': [3],     # 更大的分裂阈值
    'min_samples_leaf': [12,13,15]            # 更大的叶子样本
}

# ========== 全局变量 ==========
predictions_list = []
model_metrics = {}


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

    # 1. 严重过拟合: 训练损失极低 + 损失差距巨大
    if train_loss < 1.0 and loss_gap > 1.5:
        return {
            'status': '过拟合',
            'severity': '严重',
            'reason': f'训练损失({train_loss:.4f})极低，测试损失({test_loss:.4f})远高于训练损失, 差距={loss_gap:.4f}',
            'suggestion': '大幅减小max_depth, 增加min_samples_leaf, 减少n_estimators, 考虑增加正则化'
        }
    # 2. 中度过拟合: 损失差距明显 OR 命中率差距明显
    if loss_gap > 0.8 or acc_gap > 0.3:
        return {
            'status': '过拟合',
            'severity': '中度',
            'reason': f'训练命中率({train_acc:.4f})高于测试命中率({test_acc:.4f}), 差距={acc_gap:.4f}; 损失差距={loss_gap:.4f}',
            'suggestion': '减小max_depth, 增加min_samples_leaf/min_samples_split, 减少n_estimators'
        }
    # 3. 轻微过拟合: 命中率差距中等
    if acc_gap > 0.15:
        return {
            'status': '轻微过拟合',
            'severity': '轻度',
            'reason': f'训练命中率({train_acc:.4f})高于测试命中率({test_acc:.4f}), 差距={acc_gap:.4f}',
            'suggestion': '适当减小模型复杂度，或增加训练数据'
        }
    # 4. 欠拟合: 训练和测试损失都很高
    if train_loss > 3.0 and test_loss > 3.0:
        return {
            'status': '欠拟合',
            'severity': '严重' if train_loss > 4.0 else '中度',
            'reason': f'训练损失({train_loss:.4f})和测试损失({test_loss:.4f})都很高',
            'suggestion': '增加max_depth, 增加n_estimators, 添加更多特征, 检查数据质量'
        }
    # 5. 训练良好
    return {
        'status': '训练良好',
        'severity': '正常',
        'reason': f'训练损失({train_loss:.4f})与测试损失({test_loss:.4f})接近, 命中率差距={acc_gap:.4f}',
        'suggestion': '当前参数合适，可尝试微调'
    }


# ================= 核心处理流程 =================
def prepare_data(column_name, df, window_size, step):
    """子流程 1：数据准备与编码"""
    t_start = time.time()
    
    r1 = sliding_window_numpy(df[column_name], window_size=window_size, step=step)
    last_col_name = r1.columns[-1]
    r1 = r1.rename(columns={last_col_name: 'label'})
    
    X = r1.iloc[:, :-1]
    y_raw = r1.iloc[:, -1]

    value_counts = y_raw.value_counts()
    valid_labels = value_counts[value_counts >= 3].index
    valid_mask = y_raw.isin(valid_labels)
    
    X = X[valid_mask].reset_index(drop=True)
    y_raw = y_raw[valid_mask].reset_index(drop=True)
    
    le = LabelEncoder()
    y_encoded = le.fit_transform(y_raw)
    num_classes = len(le.classes_)
    
    elapsed = time.time() - t_start
    print(f"  ✓ 数据准备完成")
    print(f"    特征维度: {X.shape}, 类别数: {num_classes}, 有效样本: {len(X)}")
    print(f"    耗时: {elapsed:.2f}s")
    
    return X, y_encoded, le, num_classes


def train_base_model(X, y_encoded, num_classes, column_name):
    """子流程 2：基础随机森林模型训练与评估"""
    t_start = time.time()
    
    X_train, X_test, y_train, y_test = train_test_split(X, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded)
    
    model = RandomForestClassifier(
        n_estimators=50,
        max_depth=5,
        min_samples_split=20,
        min_samples_leaf=10,
        random_state=42,
        n_jobs=-1,
        verbose=0,
        oob_score=True
    )
    
    model.fit(X_train, y_train)
    
    y_pred_proba_train = model.predict_proba(X_train)
    y_pred_proba_test = model.predict_proba(X_test)
    y_pred_train = model.predict(X_train)
    y_pred_test = model.predict(X_test)
    
    all_classes = np.arange(num_classes)
    train_log_loss = log_loss(y_train, y_pred_proba_train, labels=all_classes)
    test_log_loss = log_loss(y_test, y_pred_proba_test, labels=all_classes)
    
    train_top_k_acc = top_k_accuracy(y_train, y_pred_proba_train)
    test_top_k_acc = top_k_accuracy(y_test, y_pred_proba_test)
    
    train_accuracy = accuracy_score(y_train, y_pred_train)
    test_accuracy = accuracy_score(y_test, y_pred_test)
    
    oob_score = model.oob_score_ if hasattr(model, 'oob_score_') else 'N/A'
    
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
    print(f"  {'OOB评分':<20} {str(oob_score):<15} -")
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
        'oob_score': oob_score,
        'overfit_status': overfit_analysis['status'],
        'sample_count': {'train': len(X_train), 'test': len(X_test)}
    }
    model_metrics[column_name + '_base'] = metrics
    
    return model, X_train, y_train, X_test, y_test


def tune_model(X_train, y_train, column_name):
    """子流程 3：手动网格搜索调参（避免 make_scorer 兼容性问题）"""
    t_start = time.time()
    
    kfold = KFold(n_splits=3, shuffle=True, random_state=42)
    all_params = []
    
    for n_est in param_grid['n_estimators']:
        for max_d in param_grid['max_depth']:
            for min_split in param_grid['min_samples_split']:
                for min_leaf in param_grid['min_samples_leaf']:
                    all_params.append({
                        'n_estimators': n_est,
                        'max_depth': max_d,
                        'min_samples_split': min_split,
                        'min_samples_leaf': min_leaf
                    })
    
    best_score = -1.0
    best_params = None
    best_model = None
    all_cv_scores = []
    all_train_scores = []
    
    print(f"  正在网格搜索调参 (共 {len(all_params)} 组参数)...")
    
    for idx, params in enumerate(all_params, 1):
        cv_scores = []
        train_scores = []
        
        for fold, (train_idx, val_idx) in enumerate(kfold.split(X_train, y_train)):
            X_tr = X_train.iloc[train_idx] if isinstance(X_train, pd.DataFrame) else X_train[train_idx]
            y_tr = y_train[train_idx]
            X_val = X_train.iloc[val_idx] if isinstance(X_train, pd.DataFrame) else X_train[val_idx]
            y_val = y_train[val_idx]
            
            model = RandomForestClassifier(
                n_estimators=params['n_estimators'],
                max_depth=params['max_depth'],
                min_samples_split=params['min_samples_split'],
                min_samples_leaf=params['min_samples_leaf'],
                random_state=42,
                n_jobs=-1,
                verbose=0
            )
            model.fit(X_tr, y_tr)
            
            y_pred_proba_val = model.predict_proba(X_val)
            y_pred_proba_tr = model.predict_proba(X_tr)
            
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
    
    best_model = RandomForestClassifier(
        n_estimators=best_params['n_estimators'],
        max_depth=best_params['max_depth'],
        min_samples_split=best_params['min_samples_split'],
        min_samples_leaf=best_params['min_samples_leaf'],
        random_state=42,
        n_jobs=-1,
        verbose=0,
        oob_score=True
    )
    best_model.fit(X_train, y_train)
    
    y_pred_proba_train = best_model.predict_proba(X_train)
    train_log_loss = log_loss(y_train, y_pred_proba_train, labels=best_model.classes_)
    
    importances = best_model.feature_importances_
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
    print(f"  {'OOB评分':<20} {best_model.oob_score_:<20.6f}")
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
        'oob_score': best_model.oob_score_,
        'top_features': top_features
    }
    model_metrics[column_name + '_tuned'] = metrics
    
    return best_model


def predict_and_evaluate(best_model, le, X_train_columns, df, column_name):
    """子流程 4：新数据预测与结果解码"""
    t_start = time.time()
    
    new_data = df[column_name].tail(WINDOW_SIZE - 1).tolist()
    new_data_df = pd.DataFrame([new_data], columns=X_train_columns)
    
    y_pred_proba = best_model.predict_proba(new_data_df)
    top_indices = np.argsort(y_pred_proba[0])[-TOP_PRO:]
    
    top_numbers = le.inverse_transform(top_indices)
    top_probs = y_pred_proba[0][top_indices]
    
    sorted_pairs = sorted(zip(top_numbers, top_probs), key=lambda x: x[1], reverse=True)
    
    if column_name != "Blue1":
        for num, prob in sorted_pairs:
            predictions_list.append({str(int(num)): round(float(prob), 4)})
    
    elapsed = time.time() - t_start
    
    print(f"\n  {'='*50}")
    print(f"  🔮 {column_name} 预测结果（按概率降序）")
    print(f"  {'='*50}")
    print(f"  {'MLType':<8} {'BallType':<8} {'数字':<8} {'概率':<12} {'累计概率':<12}")
    print(f"  {'-'*50}")
    cum_prob = 0.0
    for rank, (number, prob) in enumerate(sorted_pairs, 1):
        cum_prob += prob
        print(f"  {'RF':<8} {column_name:<8} {int(number):<8} {prob:<12.6f} {cum_prob:<12.6f}")
    print(f"  {'-'*50}")
    print(f"  Top-{TOP_PRO}概率总和: {cum_prob:.6f}")
    print(f"  最大概率: {sorted_pairs[0][1]:.6f}, 最小概率: {sorted_pairs[-1][1]:.6f}")
    print(f"  {'='*50}")
    print(f"  预测耗时: {elapsed:.2f}s")
    
    return sorted_pairs


# ================= 主调度函数 =================
def process_column(column_name, df, window_size, step):
    """主流程：按顺序调用各个子流程，完成单列的完整生命周期"""
    print(f"\n{'=' * 70}")
    print(f"🚀 开始处理: {column_name}")
    print(f"{'=' * 70}")
    start_time = time.time()

    X, y_encoded, le, num_classes = prepare_data(column_name, df, window_size, step)
    
    _, X_train, y_train, _, _ = train_base_model(X, y_encoded, num_classes, column_name)
    
    best_model = tune_model(X_train, y_train, column_name)
    
    predict_and_evaluate(best_model, le, X_train.columns, df, column_name)
    
    elapsed_time = time.time() - start_time
    print(f"\n{'=' * 70}")
    print(f"✅ {column_name} 处理完成, 总耗时: {elapsed_time:.2f}秒")
    print(f"{'=' * 70}")


# ================= 主程序入口 =================
if __name__ == "__main__":
    total_start = time.time()
    
    print("=" * 70)
    print(f"🌲 随机森林 双色球预测系统 - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 70)
    
    data_file = Path("C:/Users/lw25622/ML/SSQ/1.csv")
    
    t_start = time.time()
    print(f"\n📥 正在加载数据: {data_file}")
    try:
        df = pd.read_csv(data_file)
        elapsed = time.time() - t_start
        print(f"✓ 数据加载成功, 共 {len(df)} 条记录, {len(df.columns)} 列")
        print(f"  耗时: {elapsed:.2f}秒")
    except FileNotFoundError:
        print(f"✗ 错误: 未找到数据文件 {data_file}")
        exit(1)
    
    print(f"\n⚙️ 配置参数:")
    print(f"  窗口大小: {WINDOW_SIZE}, 步长: {WINDOW_STEP}, Top-K: {TOP_PRO}")
    print(f"  目标列: {TARGET_COLS}")
    print(f"  调参网格: {param_grid}")
    
    column_times = {}
    for col in TARGET_COLS:
        if col in df.columns:
            col_start = time.time()
            process_column(col, df, WINDOW_SIZE, WINDOW_STEP)
            column_times[col] = time.time() - col_start
        else:
            print(f"\n⚠ 警告: 数据集中未找到列: {col}，已跳过")
    
    print("\n" + "=" * 70)
    print("📊 全局评估报告")
    print("=" * 70)
    
    print("\n--- 各列处理耗时统计 ---")
    print(f"{'列名':<10} {'耗时(秒)':<12} {'平均耗时':<12}")
    print(f"{'-'*35}")
    total_col_time = sum(column_times.values())
    for col, t in column_times.items():
        avg_time = t / len(TARGET_COLS) if len(TARGET_COLS) > 0 else 0
        print(f"{col:<10} {t:<12.2f} {avg_time:<12.2f}")
    print(f"{'-'*35}")
    print(f"{'合计':<10} {total_col_time:<12.2f} -")
    
    print("\n--- 模型可信度分析 ---")
    print(f"{'列名':<10} {'拟合状态':<12} {'基础Top-K':<12} {'调优Top-K':<12}")
    print(f"{'-'*46}")
    for col in TARGET_COLS:
        if col in df.columns:
            base_metrics = model_metrics.get(col + '_base', {})
            tuned_metrics = model_metrics.get(col + '_tuned', {})
            status = base_metrics.get('overfit_status', 'N/A')
            base_acc = base_metrics.get('test_top_k_acc', 0)
            tuned_acc = tuned_metrics.get('cv_mean_score', 0)
            print(f"{col:<10} {status:<12} {base_acc:<12.4f} {tuned_acc:<12.4f}")
    print(f"{'-'*46}")
    
    if predictions_list:
        sorted_data = sorted(
            predictions_list,
            key=lambda x: int(list(x.keys())[0]),
            reverse=False
        )
        
        file_name = datetime.now().strftime("%Y%m%d") + ".csv"
        file_path = "./" + file_name
        
        result_df = pd.DataFrame.from_records(sorted_data, index=None).T.stack().reset_index()
        result_df = result_df.drop(columns=['level_1'])
        result_df.columns = ['number', 'probability']
        result_df.to_csv(file_path, index=False, mode='a')
        
        print(f"\n💾 预测结果已保存至: {file_path}")
    
    total_time = time.time() - total_start
    print(f"\n{'=' * 70}")
    print(f"🎉 处理完成!")
    print(f"总耗时: {total_time:.2f}秒")
    print(f"特征工程耗时: 不适用（本脚本不使用特征工程）")
    print(f"模型训练耗时: {total_col_time:.2f}秒")
    print(f"{'=' * 70}")


