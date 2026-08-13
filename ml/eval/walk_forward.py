"""双色球预测系统 - 滚动验证(walk-forward) 与基线对比

时间序列不能用随机 train_test_split（未来泄露进训练）。
改用 walk-forward：用第 1..k 期训练 → 预测第 k+1 期 → 滑窗推进 →
汇总所有真实外推期的命中率，与随机/频率基线同口径对比。
"""
from typing import Callable, Dict, List, Tuple

import numpy as np
import pandas as pd

from ml.data import load_data
from ml.config import RED_COLS, BLUE_COLS

# 红球随机选 6 号的理论期望重叠 = 6*6/33 ≈ 1.0909（超几何分布均值）
RED_RANDOM_EXPECT = 6 * 6 / 33.0
# 蓝球随机命中率理论值 = 1/16（每期独立）
BLUE_RANDOM_EXPECT = 1 / 16.0


def set_overlap(true_reds: List[int], pred_reds: List[int]) -> int:
    """红球集合命中的号码个数 (0..6)。"""
    return len(set(true_reds) & set(pred_reds))


def blue_random_baseline() -> float:
    """蓝球随机基线：每期独立 1/16。"""
    return BLUE_RANDOM_EXPECT


def random_red_baseline(n_trials: int = 2000, seed: int = 0) -> float:
    """随机选 6 号，与中奖 6 号的期望重叠（超几何分布均值 = 6*6/33）。"""
    rng = np.random.default_rng(seed)
    s = 0.0
    for _ in range(n_trials):
        pick = set(rng.choice(range(1, 34), size=6, replace=False).tolist())
        # 用真实数据每期随机抽，模拟"随机选号 vs 实际开奖"
        s += 0  # placeholder，真实基线在 run_walk_forward 内按实际开奖算
    return 6 * 6 / 33.0  # 理论期望


def run_walk_forward(
    model_factory: Callable[[pd.DataFrame], object],
    df: pd.DataFrame,
    train_min: int = 800,
    horizon: int = 120,
    step: int = 1,
    predict_fn: Callable[[object, pd.DataFrame, int], Tuple[List[int], int]] = None,
) -> Dict:
    """对单个模型做滚动外推。

    Args:
        model_factory: 输入"截至 t-1 期"的 DataFrame，返回训练好的模型对象。
        df: 全量数据(含 Red1..Red6, Blue1)。
        train_min: 最少训练期数。
        horizon: 外推多少期。
        step: 每步推进期数(默认1)。
        predict_fn: (model, df_trunc, t) -> (pred_reds:List[int], pred_blue:int)。
                    需由调用方按模型类型实现(因为旧模型/新模型接口不同)。

    Returns:
        含每期命中明细与汇总的字典。
    """
    n = len(df)
    start = train_min
    end = min(n - 1, train_min + horizon)

    red_overlaps = []          # 每期红球集合命中数
    blue_hits = []             # 每期蓝球是否命中(0/1)
    details = []

    t = start
    while t <= end:
        df_trunc = df.iloc[:t].copy()
        model = model_factory(df_trunc)
        pred_reds, pred_blue = predict_fn(model, df_trunc, t)
        true_reds = df.iloc[t][RED_COLS].astype(int).tolist()
        true_blue = int(df.iloc[t][BLUE_COLS[0]])
        ov = set_overlap(true_reds, pred_reds)
        bh = 1 if pred_blue == true_blue else 0
        red_overlaps.append(ov)
        blue_hits.append(bh)
        details.append({"t": t, "pred_reds": pred_reds, "true_reds": true_reds,
                        "overlap": ov, "pred_blue": pred_blue, "true_blue": true_blue,
                        "blue_hit": bh})
        t += step

    red_arr = np.array(red_overlaps)
    blue_arr = np.array(blue_hits)
    return {
        "n_periods": len(red_overlaps),
        "red_mean_overlap": float(red_arr.mean()),
        "red_hit_ge1": float((red_arr >= 1).mean()),
        "red_hit_ge3": float((red_arr >= 3).mean()),
        "red_hit_6": float((red_arr >= 6).mean()),
        "blue_top1_acc": float(blue_arr.mean()),
        "details": details,
    }


def freq_red_baseline_overlap(df: pd.DataFrame, start: int, end: int,
                              train_min: int, window: int = 200) -> float:
    """频率基线：用截至 t-1 的最近 window 期最频繁 6 号，算与第 t 期真实开奖的平均重叠。"""
    from collections import Counter
    ovs = []
    for t in range(start, min(end + 1, len(df))):
        c = Counter()
        lo = max(0, t - window)
        for _, row in df.iloc[lo:t].iterrows():
            for x in row[RED_COLS].astype(int):
                c[x] += 1
        top6 = [x for x, _ in c.most_common(6)]
        true = df.iloc[t][RED_COLS].astype(int).tolist()
        ovs.append(len(set(top6) & set(true)))
    return float(np.mean(ovs)) if ovs else 0.0


def random_red_overlap_period(df: pd.DataFrame, start: int, end: int,
                              n_trials: int = 1000, seed: int = 1) -> np.ndarray:
    """随机基线(数组版)：每期 MC 抽 n_trials 次随机 6 号，返回每期平均重叠数组。

    Args:
        df: 全量数据(含 Red1..Red6)。
        start: 起始期索引(含)。
        end: 结束期索引(含)。
        n_trials: 每期蒙特卡洛抽样次数(默认 1000, 与 random_red_overlap_actual 统一 N)。
        seed: 随机种子。

    Returns:
        np.ndarray，长度 = 实际外推期数，元素 = 该期 n_trials 次随机注单的平均重叠。
    """
    rng = np.random.default_rng(seed)
    ovs = []
    for t in range(start, min(end + 1, len(df))):
        true = set(df.iloc[t][RED_COLS].astype(int).tolist())
        s = 0.0
        for _ in range(n_trials):
            pick = rng.choice(range(1, 34), size=6, replace=False).tolist()
            s += len(set(pick) & true)
        ovs.append(s / n_trials)
    return np.asarray(ovs, dtype=np.float64)


def random_red_overlap_actual(df: pd.DataFrame, start: int, end: int,
                              n_trials: int = 1000, seed: int = 1) -> float:
    """随机基线：每期随机选 6 号，与真实开奖重叠，取多次平均(返回单均值)。

    默认 n_trials=1000（与 evaluate 蒙特卡洛基线统一 N）；旧调用方按位置传参不受影响。
    """
    arr = random_red_overlap_period(df, start, end, n_trials=n_trials, seed=seed)
    return float(np.mean(arr)) if len(arr) else 0.0
