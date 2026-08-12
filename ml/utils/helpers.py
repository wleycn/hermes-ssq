"""
双色球预测系统 - 通用工具函数模块
包含滑动窗口、评估指标、过拟合分析等共享逻辑
"""
import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime
import logging
import time


# ================= 滑动窗口工具 =================
def sliding_window_numpy(data_list, window_size, step=1):
    """使用 NumPy 内存视图实现高性能滑动窗口拆解
    
    Args:
        data_list: 一维数据序列
        window_size: 窗口大小
        step: 滑动步长
        
    Returns:
        pd.DataFrame: 滑动窗口视图，每行是一个窗口
    """
    arr = np.asarray(data_list)
    n = arr.shape[0]
    num_windows = (n - window_size) // step + 1
    if num_windows <= 0:
        return pd.DataFrame()
    
    strides = arr.strides[0]
    new_shape = (num_windows, window_size)
    new_strides = (strides * step, strides)
    windows = np.lib.stride_tricks.as_strided(arr, shape=new_shape, strides=new_strides)
    return pd.DataFrame(windows)


def create_sequential_windows(df, feature_cols, window_size):
    """创建序列模型的滑动窗口数据集（3D输入: [batch, seq_len, features]）
    
    Args:
        df: 包含所有特征的 DataFrame
        feature_cols: 特征列名列表
        window_size: 时间窗口大小
        
    Returns:
        tuple: (X, y) 其中 X 形状为 (n_samples, window_size, n_features)
    """
    n = len(df)
    X_data = df[feature_cols].values
    n_features = len(feature_cols)
    
    X = np.zeros((n - window_size, window_size, n_features), dtype=np.float32)
    
    for i in range(n - window_size):
        X[i] = X_data[i:i + window_size]
    
    return X


# ================= 评估指标 =================
def top_k_accuracy(y_true, y_pred_proba, k=6):
    """计算 Top-K 命中率（向量化高性能版）
    
    Args:
        y_true: 真实标签数组
        y_pred_proba: 预测概率数组 (n_samples, n_classes)
        k: 前K个候选
        
    Returns:
        float: Top-K 命中率
    """
    if y_pred_proba.ndim < 2 or y_pred_proba.shape[1] < 2:
        return 0.0
    top_k = min(k, y_pred_proba.shape[1])
    top_indices = np.argsort(y_pred_proba, axis=1)[:, -top_k:]
    correct = np.any(top_indices == y_true.reshape(-1, 1), axis=1)
    return float(np.mean(correct))


def top_k_multi_label_accuracy(y_true_proba, y_pred_proba, k=6):
    """多标签 Top-K 命中率（适用于红球6个位置）
    
    Args:
        y_true_proba: 真实标签 multi-hot (n_samples, n_classes)
        y_pred_proba: 预测概率 (n_samples, n_classes)
        k: 前K个候选
        
    Returns:
        float: 平均 Top-K 命中率
    """
    total_correct = 0.0
    n_samples = len(y_true_proba)
    
    for i in range(n_samples):
        top_indices = np.argsort(y_pred_proba[i])[-k:]
        true_indices = np.where(y_true_proba[i] == 1)[0]
        total_correct += len(set(top_indices) & set(true_indices)) / len(true_indices)
    
    return total_correct / n_samples if n_samples > 0 else 0.0


def evaluate_prediction_coverage(y_true, y_pred_proba, k_list=[3, 5, 6, 10]):
    """评估不同K值下的命中率覆盖
    
    Args:
        y_true: 真实标签
        y_pred_proba: 预测概率
        k_list: 要评估的K值列表
        
    Returns:
        dict: 各K值对应的命中率
    """
    results = {}
    for k in k_list:
        results[f'top_{k}'] = top_k_accuracy(y_true, y_pred_proba, k=k)
    return results


# ================= 过拟合分析 =================
def analyze_overfitting(train_loss, test_loss, train_acc, test_acc):
    """分析模型是否过拟合/欠拟合（分级判断）
    
    Args:
        train_loss: 训练损失
        test_loss: 测试损失
        train_acc: 训练命中率
        test_acc: 测试命中率
        
    Returns:
        dict: 分析结果包含 status, severity, reason, suggestion
    """
    loss_gap = test_loss - train_loss
    acc_gap = train_acc - test_acc

    # 严重过拟合
    if train_loss < 1.0 and loss_gap > 1.5:
        return {
            'status': '过拟合',
            'severity': '严重',
            'reason': f'训练损失({train_loss:.4f})极低，测试损失({test_loss:.4f})远高于训练损失, 差距={loss_gap:.4f}',
            'suggestion': '大幅减小模型复杂度，增加正则化，减少训练轮数'
        }
    # 中度过拟合
    if loss_gap > 0.8 or acc_gap > 0.3:
        return {
            'status': '过拟合',
            'severity': '中度',
            'reason': f'训练命中率({train_acc:.4f})高于测试命中率({test_acc:.4f}), 差距={acc_gap:.4f}; 损失差距={loss_gap:.4f}',
            'suggestion': '减小模型复杂度，增加正则化强度'
        }
    # 轻微过拟合
    if acc_gap > 0.15:
        return {
            'status': '轻微过拟合',
            'severity': '轻度',
            'reason': f'训练命中率({train_acc:.4f})高于测试命中率({test_acc:.4f}), 差距={acc_gap:.4f}',
            'suggestion': '适当减小模型复杂度，或增加训练数据'
        }
    # 欠拟合
    if train_loss > 3.0 and test_loss > 3.0:
        return {
            'status': '欠拟合',
            'severity': '严重' if train_loss > 4.0 else '中度',
            'reason': f'训练损失({train_loss:.4f})和测试损失({test_loss:.4f})都很高',
            'suggestion': '增加模型复杂度，添加更多特征'
        }
    # 训练良好
    return {
        'status': '训练良好',
        'severity': '正常',
        'reason': f'训练损失({train_loss:.4f})与测试损失({test_loss:.4f})接近, 命中率差距={acc_gap:.4f}',
        'suggestion': '当前参数合适，可尝试微调'
    }


# ================= 日志工具 =================
def setup_logger(name, log_file=None, level=logging.INFO):
    """配置日志记录器
    
    Args:
        name: 日志记录器名称
        log_file: 日志文件路径（可选）
        level: 日志级别
        
    Returns:
        logging.Logger: 配置好的日志记录器
    """
    logger = logging.getLogger(name)
    logger.setLevel(level)
    
    # 控制台处理器
    if not logger.handlers:
        console_handler = logging.StreamHandler()
        console_handler.setLevel(level)
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)
        
        # 文件处理器
        if log_file:
            file_handler = logging.FileHandler(log_file, encoding='utf-8')
            file_handler.setLevel(level)
            file_handler.setFormatter(formatter)
            logger.addHandler(file_handler)
    
    return logger


# ================= 结果保存工具 =================
def save_prediction_results(predictions, output_dir, model_name):
    """保存预测结果到CSV文件
    
    Args:
        predictions: 预测结果列表
        output_dir: 输出目录
        model_name: 模型名称
    """
    today = datetime.now().strftime("%Y%m%d")
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    output_file = output_dir / f"{model_name}_{today}.csv"
    
    if isinstance(predictions, pd.DataFrame):
        predictions.to_csv(output_file, index=False, encoding='utf-8-sig')
    else:
        pd.DataFrame(predictions).to_csv(output_file, index=False, encoding='utf-8-sig')
    
    print(f"预测结果已保存至: {output_file}")
    return str(output_file)


def build_prediction_dataframe(red_probs, blue_probs=None, red_range=range(1, 34), blue_range=range(1, 17)):
    """构建预测结果DataFrame
    
    Args:
        red_probs: 红球概率数组
        blue_probs: 蓝球概率数组（可选）
        red_range: 红球号码范围
        blue_range: 蓝球号码范围
        
    Returns:
        pd.DataFrame: 格式化的预测结果
    """
    records = []
    
    for i, num in enumerate(red_range):
        records.append({
            'Type': 'Red',
            'Number': num,
            'Probability': red_probs[i],
            'Rank': np.argsort(red_probs)[::-1].tolist().index(i) + 1
        })
    
    if blue_probs is not None:
        for i, num in enumerate(blue_range):
            records.append({
                'Type': 'Blue',
                'Number': num,
                'Probability': blue_probs[i],
                'Rank': np.argsort(blue_probs)[::-1].tolist().index(i) + 1
            })
    
    result_df = pd.DataFrame(records)
    result_df = result_df.sort_values(['Type', 'Probability'], ascending=[True, False]).reset_index(drop=True)
    return result_df


def print_banner(text, char="=", length=60):
    """打印分隔横幅
    
    Args:
        text: 横幅文字
        char: 填充字符
        length: 横幅长度
    """
    print(f"\n{char * length}")
    print(f"  {text}")
    print(f"{char * length}\n")


def print_model_report(metrics, model_name="模型"):
    """打印模型评估报告
    
    Args:
        metrics: 指标字典
        model_name: 模型名称
    """
    print(f"\n{'='*50}")
    print(f"  📊 {model_name} 评估报告")
    print(f"{'='*50}")
    
    for key, value in metrics.items():
        if isinstance(value, float):
            print(f"  {key:<30} {value:.6f}")
        elif isinstance(value, dict):
            print(f"  {key}:")
            for sub_key, sub_val in value.items():
                print(f"    {sub_key:<28} {sub_val}")
        else:
            print(f"  {key:<30} {value}")
    
    print(f"{'='*50}")