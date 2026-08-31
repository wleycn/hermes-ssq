"""变化点检测模块 (ml/changepoint.py) —— CP-ADAPT Layer A。

信仰模式约定:
  - 不预设序列 i.i.d.; 看检测器在真实数据上的输出。
  - 但不在"开奖号码本身"上做 CP 检测(那必然噪声爆表), 而在"模型残差/滚动表现"上做:
    主统计量 = 各模型历史每期命中数(overlap)的滚动序列。
    若命中率在某期后系统性偏移, 才提示"规律变了"。信噪比远高于裸号码。

非侵入契约:
  - 默认(CP_ADAPT.enabled=False) 所有调用方走 else 原分支, 本模块不被调用。
  - ON 时产物隔离: 训练截断/衰减只在调用方显式启用; 本模块纯函数, 无副作用。

方法:
  - 滑窗双样本 KS 检验: 以当前候选分割点 split 把序列分为左/右两段,
    比较两段分布是否显著不同; 扫描所有合法 split, 取最优 p 值。
  - 仅在 p < p_threshold 且两段均 >= min_window 时确认变点。
  - 输出: detect_changepoints -> List[int](期索引, 0-based), 以及 current_regime_start。
"""
from __future__ import annotations

import numpy as np
from scipy.stats import ks_2samp

from ml.config import CP_ADAPT


def detect_changepoint_in_series(
    series: np.ndarray,
    min_window: int | None = None,
    p_threshold: float | None = None,
    lookback_max: int | None = None,
) -> tuple[list[int], float]:
    """在单条滚动统计量序列上扫描最优变点。

    Args:
        series: 1-D 滚动统计量序列(如每期命中数), 长度 L。
        min_window: 单侧最少期数。
        p_threshold: 显著性阈值(越小越严)。
        lookback_max: 检测用最大滚动窗口(从序列尾部向前看的最远期数)。

    Returns:
        (changepoints, best_p):
          changepoints: 变点索引列表(0-based, 即 split 点, 右段起点)。
          best_p: 该变点对应的 KS p 值(无变点时 None)。
    """
    min_window = min_window or CP_ADAPT["min_window"]
    p_threshold = p_threshold if p_threshold is not None else CP_ADAPT["p_threshold"]
    if lookback_max is None:
        lookback_max = CP_ADAPT["lookback_max"]  # 显式 None 也回退到 config
    L = len(series)
    if L < 2 * min_window:
        return [], None  # 样本不足, 不判
    if lookback_max is None or lookback_max >= L:
        # 全史检测: 从最小窗口起点开始, 不丢弃任何样本(样本少时更具代表性)
        start = min_window
    else:
        # 仅看最近 lookback_max 期(更早的变点对当前机制无决策价值)
        start = max(min_window, L - lookback_max)
    best_split: int | None = None
    best_p: float = 1.0
    for split in range(start, L - min_window + 1):
        left = series[:split]
        right = series[split:]
        if len(left) < min_window or len(right) < min_window:
            continue
        res = ks_2samp(left, right)
        p: float = float(res.pvalue)
        if p < best_p:
            best_p = p
            best_split = split
    if best_split is not None and best_p < p_threshold:
        return [best_split], best_p
    return [], None


def detect_changepoints(
    overlap_matrix: np.ndarray,
    min_window: int | None = None,
    p_threshold: float | None = None,
    lookback_max: int | None = None,
) -> dict:
    """对多模型命中矩阵(每期每模型 overlap)做变点检测, 聚合出"当前机制起点"。

    Args:
        overlap_matrix: shape (n_periods, n_models) 的命中数矩阵(整数)。
        (其余参数同 detect_changepoint_in_series)

    Returns:
        {
          "per_model": {midx: {"cp": [...], "p": float}},
          "consensus_splits": [int],      # 多数模型共同指向的变点(>=半数模型一致)
          "current_regime_start": int,    # 最近一个共识变点之后(无则 0)
          "series": overlap_matrix,       # 回显, 便于调用方核对
        }
    """
    if overlap_matrix.size == 0:
        return {
            "per_model": {},
            "consensus_splits": [],
            "current_regime_start": 0,
            "series": overlap_matrix,
        }
    n_periods, n_models = overlap_matrix.shape
    per_model: dict = {}
    split_votes: dict[int, int] = {}
    for m in range(n_models):
        cps, p = detect_changepoint_in_series(
            overlap_matrix[:, m], min_window, p_threshold, lookback_max
        )
        per_model[m] = {"cp": cps, "p": p}
        for c in cps:
            split_votes[c] = split_votes.get(c, 0) + 1

    threshold = max(1, n_models // 2)  # 半数以上模型一致才算共识
    consensus = sorted([s for s, v in split_votes.items() if v >= threshold])
    current_regime_start = consensus[-1] if consensus else 0
    return {
        "per_model": per_model,
        "consensus_splits": consensus,
        "current_regime_start": int(current_regime_start),
        "series": overlap_matrix,
    }


def regime_weights(
    n_periods: int,
    current_regime_start: int,
    halflife: int | None = None,
) -> np.ndarray:
    """生成 regime 指数衰减权重(近权重高, 远权重低)。

    用于 --regime-window N: 不硬截断, 而是对全部历史加权,
    当前机制内权重~1, 旧机制按半衰期衰减。

    Args:
        n_periods: 总期数。
        current_regime_start: 当前机制起点索引(0-based)。
        halflife: 半衰期(期)。

    Returns:
        shape (n_periods,) 的权重, 末位=1.0。
    """
    halflife = halflife or CP_ADAPT["decay_halflife"]
    idx = np.arange(n_periods)
    age = idx - current_regime_start  # <0 表示在变点之前(旧机制)
    # 变点之后(age>=0): 权重 1; 变点之前: 按半衰期衰减
    w = np.where(age >= 0, 1.0, np.exp(age / halflife * np.log(0.5)))
    return w.astype(float)


def selfcheck() -> dict:
    """自检: 注入已知突变应检出, 平稳序列应不报。

    Returns: {"mutated_detected": bool, "stationary_clean": bool, "details": ...}
    """
    rng = np.random.default_rng(0)
    # 场景1: 明显突变 — 前300期均值1.0, 后300期均值3.0 (overlap 分布整体右移)
    s1 = np.concatenate([rng.poisson(1.0, 300), rng.poisson(3.0, 300)]).astype(float)
    cps1, p1 = detect_changepoint_in_series(s1, min_window=50)
    mutated_detected = len(cps1) == 1 and 250 <= cps1[0] <= 350

    # 场景2: 平稳 — 全程 Poisson(2.0), 不应报变点
    s2 = rng.poisson(2.0, 600).astype(float)
    cps2, p2 = detect_changepoint_in_series(s2, min_window=50)
    stationary_clean = len(cps2) == 0

    # 场景3: 多模型共识 — 3/3 模型都在同一 split 突变
    m = np.vstack([
        np.concatenate([rng.poisson(1.0, 300), rng.poisson(3.0, 300)]),
        np.concatenate([rng.poisson(1.0, 300), rng.poisson(3.0, 300)]),
        np.concatenate([rng.poisson(1.0, 300), rng.poisson(3.0, 300)]),
    ]).T.astype(float)
    res = detect_changepoints(m)
    consensus_ok = res["consensus_splits"] and 250 <= res["consensus_splits"][-1] <= 350

    return {
        "mutated_detected": mutated_detected,
        "stationary_clean": stationary_clean,
        "consensus_ok": consensus_ok,
        "p_mutated": p1,
        "p_stationary": p2,
        "all_pass": mutated_detected and stationary_clean and consensus_ok,
    }


if __name__ == "__main__":
    out = selfcheck()
    print("CP-ADAPT selfcheck:")
    for k, v in out.items():
        print(f"  {k}: {v}")
    raise SystemExit(0 if out["all_pass"] else 1)
