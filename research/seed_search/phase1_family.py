#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Phase 1 族级搜索(H1): 四常数切片种子族 vs 随机对照。

口径:
  - 候选池 = 全域均匀(33 球无放回), 与 Phase 0 内核一致。
  - 常数: π / e / √2 / φ, 各算 100万+5 位小数, d=6 位切片步长1 → 每族 100万种子。
  - 对照: 100万 × 4 = 400万随机种子(均匀 [0, 999999], 与切片同范围)。
  - 得分 = red_mean_hits(全历史窗口, 等价性定理下最稳定)。
  - 族级检验: 双样本均值 z + KS + max 对比, Bonferroni α=0.05/4=0.0125。
"""
from __future__ import annotations

import json
import multiprocessing as mp
import sys
import time
from pathlib import Path

PROJ = Path("/home/hermes/workspace/python/SSQ")
sys.path.insert(0, str(PROJ))

import numpy as np  # noqa: E402

from ml.pg_conn import connect  # noqa: E402
from research.seed_search.phase0_kernel import build_freq, load_draws  # noqa: E402

OUT = Path("/home/hermes/.hermes/cache/scratch/seed_p1_family.json")
N_PER_FAMILY = 1_000_000
N_CONTROL = 4_000_000
D = 6

# 全局(worker 用): 全历史频率
_P_RED: np.ndarray | None = None
_P_BLUE: np.ndarray | None = None


def _worker_init(p_red: np.ndarray, p_blue: np.ndarray) -> None:
    global _P_RED, _P_BLUE
    _P_RED = p_red
    _P_BLUE = p_blue


def _score_batch(seeds: np.ndarray) -> np.ndarray:
    """批量算 red_mean_hits(全历史)。纯 numpy, 每种子 default_rng。"""
    out = np.empty(len(seeds), dtype=np.float64)
    p_red = _P_RED
    for i, s in enumerate(seeds):
        rng = np.random.default_rng(int(s))
        reds = np.stack([rng.choice(33, size=6, replace=False) for _ in range(30)])
        out[i] = float(p_red[reds].sum() / 30)
    return out


def _worker(args) -> np.ndarray:
    seeds, = args
    return _score_batch(seeds)


def constant_seeds(constant: str, n: int, d: int) -> np.ndarray:
    """常数小数部分 d 位切片步长1 → n 个种子。"""
    import mpmath
    mpmath.mp.dps = n + d + 2
    if constant == "pi":
        s = str(mpmath.pi)
    elif constant == "e":
        s = str(mpmath.e)
    elif constant == "sqrt2":
        s = str(mpmath.sqrt(2))
    elif constant == "phi":
        s = str(mpmath.mp.mpf((1 + mpmath.mp.sqrt(5)) / 2))
    else:
        raise ValueError(constant)
    frac = s.split(".")[1]
    seeds = np.array([int(frac[i:i + d]) for i in range(n)], dtype=np.int64)
    return seeds


def main() -> None:
    conn = connect()
    try:
        reds, blues = load_draws(conn)
    finally:
        conn.close()
    p_red, p_blue, n_draws = build_freq(reds, blues, window=None)

    t0 = time.perf_counter()
    families: dict = {}
    for name in ("pi", "e", "sqrt2", "phi"):
        seeds = constant_seeds(name, N_PER_FAMILY, D)
        families[name] = seeds
        print(f"[seed] {name}: {len(seeds)} seeds, unique={len(np.unique(seeds))}, "
              f"range=[{seeds.min()},{seeds.max()}]", flush=True)

    rng = np.random.default_rng(20261125)
    control = rng.integers(0, 10 ** D - 1, size=N_CONTROL)
    print(f"[seed] control: {len(control)} seeds", flush=True)

    # ---- 多进程算分 ----
    n_proc = min(8, mp.cpu_count())
    chunks: list[np.ndarray] = []
    for name in ("pi", "e", "sqrt2", "phi"):
        seeds = families[name]
        for i in range(0, len(seeds), 200_000):
            chunks.append(seeds[i:i + 200_000])
    for i in range(0, len(control), 200_000):
        chunks.append(control[i:i + 200_000])

    print(f"[run] {len(chunks)} chunks × {n_proc} procs", flush=True)
    with mp.Pool(n_proc, initializer=_worker_init, initargs=(p_red, p_blue)) as pool:
        results = pool.map(_worker, [(c,) for c in chunks], chunksize=1)

    # 结果归属: 前 4×5=20 个 chunk 是四族(每族 5 chunk), 后 20 个是对照
    n_fam_chunks = len(chunks) - (N_CONTROL // 200_000)
    fam_scores: dict = {}
    idx = 0
    for name in ("pi", "e", "sqrt2", "phi"):
        fam_scores[name] = np.concatenate(results[idx:idx + 5])
        idx += 5
    control_scores = np.concatenate(results[n_fam_chunks:])

    # ---- 族级比较 ----
    def summary(scores: np.ndarray) -> dict:
        return {
            "mean": round(float(scores.mean()), 7),
            "std": round(float(scores.std()), 7),
            "max": round(float(scores.max()), 7),
            "q95": round(float(np.quantile(scores, 0.95)), 7),
            "q99": round(float(np.quantile(scores, 0.99)), 7),
        }

    from math import sqrt
    c_mean = float(control_scores.mean())
    c_std = float(control_scores.std())
    c_max = float(control_scores.max())
    out: dict = {
        "n_draws": n_draws,
        "n_per_family": N_PER_FAMILY,
        "n_control": N_CONTROL,
        "control": summary(control_scores),
        "families": {},
        "family_tests": [],
    }
    alpha_bonf = 0.05 / 4
    for name in ("pi", "e", "sqrt2", "phi"):
        s = fam_scores[name]
        sm = summary(s)
        out["families"][name] = sm
        # 双样本 z
        se = sqrt(c_std ** 2 / len(control_scores) + float(s.std()) ** 2 / len(s))
        z = (float(s.mean()) - c_mean) / se
        # KS
        from scipy import stats as sps
        ks = sps.ks_2samp(s, control_scores)
        out["family_tests"].append({
            "family": name,
            "mean_z_vs_control": round(z, 3),
            "mean_sig": bool(abs(z) > 2.576),  # Bonferroni 双侧
            "ks_stat": round(float(ks.statistic), 5),
            "ks_p": round(float(ks.pvalue), 5),
            "ks_sig": bool(ks.pvalue < alpha_bonf),
            "max_vs_control": round(float(s.max()) - c_max, 6),
        })

    out["elapsed_sec"] = round(time.perf_counter() - t0, 1)
    with open(OUT, "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)
    print(json.dumps(out, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
