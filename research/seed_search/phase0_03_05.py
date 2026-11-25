#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Phase 0.3 空分布标定(多窗口) + 0.5 相关种子自相关。

0.3: 对每个窗口(全历史/500/100/30期), 跑 N 个随机种子, 记录
     red_mean_hits 分布: 均值/σ/max/max-z, 与理论极值 E[max]≈μ+σ√(2lnN) 核对。
0.5: π 连续切片种子(314159..314258, 100个)的 red_mean_hits 滞后1自相关,
     验证 default_rng 对连续种子的输出流独立性(Binh Ho Remarks 警告点)。
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

PROJ = Path("/home/hermes/workspace/python/SSQ")
sys.path.insert(0, str(PROJ))

import numpy as np  # noqa: E402

from ml.pg_conn import connect  # noqa: E402
from research.seed_search.phase0_kernel import (  # noqa: E402
    RED_BASE, build_freq, load_draws, seed_to_score)

OUT = Path("/home/hermes/.hermes/cache/scratch/seed_p0_03_05.json")
N_NULL = 100_000   # 空分布种子数(计划 0.3 量级)


def null_dist(rng: np.random.Generator, p_red: np.ndarray, p_blue: np.ndarray,
              n_seeds: int) -> dict:
    """跑 n_seeds 个随机种子, 返回 red_mean_hits 的均值/σ/max/max-z。"""
    seeds = rng.integers(0, 2**31 - 1, size=n_seeds)
    reds = np.array([seed_to_score(int(s), p_red, p_blue)[0] for s in seeds])
    mean = float(reds.mean())
    std = float(reds.std())
    mx = float(reds.max())
    max_z = (mx - mean) / std if std > 0 else 0.0
    theory_max_z = float(np.sqrt(2 * np.log(n_seeds)))
    return {
        "n": n_seeds,
        "mean": round(mean, 6),
        "std": round(std, 6),
        "max": round(mx, 6),
        "max_z": round(max_z, 3),
        "theory_max_z": round(theory_max_z, 3),
        "lucky_king_excess": round(mx - RED_BASE, 5),
    }


def main() -> None:
    conn = connect()
    try:
        reds, blues = load_draws(conn)
    finally:
        conn.close()

    out: dict = {}
    master = np.random.default_rng(20261125)

    # ---- 0.3 空分布: 多窗口 ----
    null_by_window: dict = {}
    for window in (None, 500, 100, 30):
        p_red, p_blue, n = build_freq(reds, blues, window=window)
        label = "all" if window is None else f"w{window}"
        null_by_window[label] = null_dist(master, p_red, p_blue, N_NULL)
    out["null_dist"] = null_by_window

    # ---- 0.5 相关种子自相关: π 连续切片 ----
    p_red_all, p_blue_all, _ = build_freq(reds, blues, window=None)
    base = 314159
    n_seq = 200
    seq_scores = np.array([
        seed_to_score(base + i, p_red_all, p_blue_all)[0] for i in range(n_seq)
    ])
    m = seq_scores.mean()
    # 滞后 1 自相关
    num = float(np.sum((seq_scores[:-1] - m) * (seq_scores[1:] - m)))
    den = float(np.sum((seq_scores - m) ** 2))
    lag1 = num / den if den > 0 else 0.0
    # 连续种子注单重叠率: 相邻两种子第 1 注的号码交集(期望 6*6/33≈1.09)
    from research.seed_search.phase0_kernel import seed_to_tickets
    overlaps: list[int] = []
    for i in range(1, 50):
        r0, _ = seed_to_tickets(base + i - 1)
        r1, _ = seed_to_tickets(base + i)
        # 30 注逐注对齐重叠(第 1 注 vs 第 1 注)
        overlaps.append(len(set(int(x) for x in r0[0]) & set(int(x) for x in r1[0])))
    out["autocorr"] = {
        "n_consecutive": n_seq,
        "seed_base": base,
        "red_lag1_autocorr": round(lag1, 4),
        "noise_bound_2sd": round(2 / np.sqrt(n_seq), 4),
        "independent_if_below": bool(abs(lag1) < 2 / np.sqrt(n_seq)),
        "adjacent_note_overlap_mean": round(float(np.mean(overlaps)), 4),
        "overlap_expect": round(6 * 6 / 33, 4),
    }

    with open(OUT, "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)
    print(json.dumps(out, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
