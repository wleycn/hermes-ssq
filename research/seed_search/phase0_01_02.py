#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Phase 0.1 内核正确性 + 0.2 吞吐实测。

0.1: 验证"频率求和恒等式" = "逐期对齐"(精确相等), 且随机种子得分 ≈ 理论基线。
0.2: 1000 种子计时, 外推 1M/10M/100M 成本表。
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

PROJ = Path("/home/hermes/workspace/python/SSQ")
sys.path.insert(0, str(PROJ))

import numpy as np  # noqa: E402

from ml.pg_conn import connect  # noqa: E402
from research.seed_search.phase0_kernel import (  # noqa: E402
    RED_BASE, BLUE_BASE, build_freq, load_draws, seed_to_score)

OUT = Path("/home/hermes/.hermes/cache/scratch/seed_p0_01_02.json")


def brute_align(seed: int, reds_draws: np.ndarray, blues_draws: np.ndarray,
                n_notes: int = 30) -> tuple[float, float]:
    """慢版: 种子生成 30 注, 逐期逐注对齐, 算 red_mean_hits / blue_hit_rate。"""
    rng = np.random.default_rng(seed)
    red_notes = [set(int(x) for x in rng.choice(33, size=6, replace=False))
                 for _ in range(n_notes)]
    blue_notes = [int(rng.integers(0, 16)) for _ in range(n_notes)]
    red_hits: list[int] = []
    blue_hits: list[int] = []
    for t in range(reds_draws.shape[0]):
        dr = set(int(x) for x in reds_draws[t])
        db = int(blues_draws[t])
        for rn in red_notes:
            red_hits.append(len(rn & dr))
        for bn in blue_notes:
            blue_hits.append(1 if bn == db else 0)
    return float(np.mean(red_hits)), float(np.mean(blue_hits))


def main() -> None:
    conn = connect()
    try:
        reds, blues = load_draws(conn)
    finally:
        conn.close()
    p_red, p_blue, n = build_freq(reds, blues, window=None)
    out: dict = {"n_draws": n}

    # ---- 0.1 恒等式验证: 频率求和 vs 逐期对齐, 5 个种子 ----
    iden: list[dict] = []
    for seed in (42, 2026094, 314159, 271828, 141421):
        fast_r, fast_b = seed_to_score(seed, p_red, p_blue)
        slow_r, slow_b = brute_align(seed, reds, blues)
        iden.append({
            "seed": seed,
            "fast_red": round(fast_r, 8),
            "slow_red": round(slow_r, 8),
            "red_diff": round(abs(fast_r - slow_r), 12),
            "fast_blue": round(fast_b, 8),
            "slow_blue": round(slow_b, 8),
            "blue_diff": round(abs(fast_b - slow_b), 12),
        })
    max_red_diff = max(d["red_diff"] for d in iden)
    out["identity_check"] = {
        "max_red_diff": max_red_diff,
        "identity_holds": max_red_diff < 1e-9,
        "details": iden,
    }

    # ---- 0.1 理论基线核对: 随机种子得分应 ≈ RED_BASE ----
    rng = np.random.default_rng(0)
    seeds = rng.integers(0, 2**31 - 1, size=2000)
    scores = np.array([seed_to_score(int(s), p_red, p_blue)[0] for s in seeds])
    out["baseline_check"] = {
        "n_random_seeds": len(seeds),
        "mean_red": round(float(scores.mean()), 6),
        "theoretical": round(RED_BASE, 6),
        "mean_blue": round(float(np.mean([seed_to_score(int(s), p_red, p_blue)[1]
                                          for s in seeds])), 6),
        "blue_theoretical": round(BLUE_BASE, 6),
        "red_std": round(float(scores.std()), 6),
    }

    # ---- 0.2 吞吐 ----
    t0 = time.perf_counter()
    n_timing = 1000
    for s in seeds[:n_timing]:
        seed_to_score(int(s), p_red, p_blue)
    dt = time.perf_counter() - t0
    per_seed = dt / n_timing
    out["throughput"] = {
        "n_timed": n_timing,
        "seconds": round(dt, 3),
        "sec_per_seed": round(per_seed, 6),
        "seeds_per_sec": round(1 / per_seed, 1),
        "cost_1M_min": round(per_seed * 1e6 / 60, 2),
        "cost_10M_min": round(per_seed * 1e7 / 60, 2),
        "cost_100M_min": round(per_seed * 1e8 / 60, 2),
    }

    with open(OUT, "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)
    print(json.dumps(out, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
