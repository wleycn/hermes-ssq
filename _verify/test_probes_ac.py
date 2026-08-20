#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""A/C 探针套件冒烟测试 + 真实数据 FLAT 证据生成。

运行：.venv/bin/python -m pytest _verify/test_probes_ac.py -s -q
或：  .venv/bin/python _verify/test_probes_ac.py   （直接跑，打印证据）
"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import numpy as np

from ml.probes.surrogate_probe import make_surrogates, surrogate_zscore, run_surrogate_probe
from ml.probes.nist_probe import encode_natural_series, run_nist_subset, summarize
from ml.probes.ordinal_probe import run_ordinal_probe, permutation_entropy, amigo_chi2_iid
from ml.probes.transfer_entropy import run_transfer_entropy_matrix, summarize_te
from ml.conformal.conformal_predict import build_from_history, summarize_coverage
from ml.conformal.edl_probe import run_edl_experiment, summarize_edl


# ----------------------------- 合成随机数据（行为校验） ----------------------------- #

def test_surrogate_on_random():
    rng = np.random.default_rng(0)
    x = rng.integers(1, 34, size=300)
    surros = make_surrogates(x, kind="aaft", n=200, seed=1)
    assert len(surros) == 200
    z = surrogate_zscore(x, np.mean, surros)
    assert abs(z) < 3  # 随机序列应落在 surrogate 分布内


def test_nist_random_bits():
    rng = np.random.default_rng(1)
    bits = rng.integers(0, 2, size=200000)  # 足量满足 NIST
    res = run_nist_subset(bits)
    assert all(r.verdict != "INVALID" for r in res)
    # 随机位流应全部 RANDOM（p>0.01）
    assert all(r.verdict == "RANDOM" for r in res if r.p is not None)


def test_ordinal_random():
    rng = np.random.default_rng(2)
    x = rng.integers(1, 34, size=2000)
    pe = permutation_entropy(x, 3)
    assert 0.9 <= pe <= 1.05  # 随机序列排列熵≈1
    r = amigo_chi2_iid(x, 3)
    assert r["verdict"] == "RANDOM"  # 随机 → 不显著


def test_te_random_independent():
    rng = np.random.default_rng(3)
    series = {f"R{i}": rng.integers(1, 34, size=500) for i in range(6)}
    res = run_transfer_entropy_matrix(series, bins=8)
    summ = summarize_te(res)
    assert summ["overall"] == "INDEPENDENT"


def test_conformal_coverage():
    rng = np.random.default_rng(4)
    n_hist = 300
    red_hist = [rng.dirichlet(np.ones(33) * 5) for _ in range(n_hist)]
    blue_hist = [rng.dirichlet(np.ones(16) * 5) for _ in range(n_hist)]
    red_draws = [rng.choice(33, 6, replace=False) + 1 for _ in range(n_hist)]
    blue_draws = [rng.integers(1, 17) for _ in range(n_hist)]
    cs = build_from_history(red_hist, blue_hist, red_draws, blue_draws, alpha=0.90)
    # 校准集经验覆盖率（每个号码单独判是否落入集合）
    red_cov = np.mean([1 if (d in cs["red"].predict_set(red_hist[i]))
                       else 0 for i in range(n_hist) for d in red_draws[i]])
    assert 0.80 <= red_cov <= 0.98


# ----------------------------- 真实数据 FLAT 证据 ----------------------------- #

def _load_real():
    import pandas as pd
    df = pd.read_csv(ROOT / "ml/data/1.csv")
    reds = df[["Red1", "Red2", "Red3", "Red4", "Red5", "Red6"]].values.tolist()
    blues = df["Blue1"].astype(int).tolist()
    return reds, blues


def run_real_evidence():
    reds, blues = _load_real()
    n = len(reds)
    print(f"\n=== 真实 SSQ 数据：{n} 期 ===")

    # A1 surrogate（用红球每期首个号码做序列 + 均值统计）
    red_first = [r[0] for r in reds]
    blue_seq = blues
    stats = {"mean": np.mean, "std": np.std, "max": np.max}
    out = run_surrogate_probe(red_first, stats, kinds=("rs", "aaft", "iaaft"), n=300, seed=7)
    print("\n[A1 Surrogate/NIST 元验证] 红球首号序列 × 3 统计量 × 3 surrogate：")
    for sname, by_kind in out.items():
        for k, v in by_kind.items():
            print(f"  {sname:5s}/{k:6s}: z={v['z']:+.3f} p={v['p']:.3f} -> {v['verdict']}")

    # A2 NIST（红球每期 6 号 ×6bit + 蓝球 ×5bit 拼成 bit 流）
    red_bits = encode_natural_series([v for r in reds for v in r], width=6)
    blue_bits = encode_natural_series(blue_seq, width=5)
    bits = np.concatenate([red_bits, blue_bits])
    nist_res = run_nist_subset(bits)
    summ = summarize(nist_res)
    print(f"\n[A2 NIST 适用子集] 总比特 {bits.size}（caveat：远低于 NIST 推荐 100,000）：")
    for r in nist_res:
        print(f"  {r.name:14s}: p={r.p} -> {r.verdict} ({r.power})")
    print(f"  汇总: {summ['overall']} | {summ['caveat']}")

    # A3 ordinal（红球首号序列）
    ord_res = run_ordinal_probe(red_first, dims=(3, 4, 5))
    print("\n[A3 Ordinal/排列熵] 红球首号序列：")
    for r in ord_res:
        print(f"  {r.name:10s}: {r.metric}={r.value:.4f} -> {r.verdict} ({r.detail})")

    # A4 transfer entropy（6 红位 + 蓝）
    series = {f"Red{i+1}": [int(r[i]) for r in reds] for i in range(6)}
    series["Blue1"] = blues
    te_res = run_transfer_entropy_matrix(series, bins=8)
    te_summ = summarize_te(te_res)
    print("\n[A4 Transfer Entropy] 跨球位有向依赖：")
    print(f"  {te_summ['overall']} | max_te={te_summ['max_te']}")
    print(f"  {te_summ['interpretation']}")

    # C2 EDL 先验实验
    edl_res = run_edl_experiment([v for r in reds for v in r], blues, window=20)
    edl_summ = summarize_edl(edl_res)
    print("\n[C2 EDL 区分度先验实验]：")
    for r in edl_res:
        print(f"  {r.ball_type}: mean_ev={r.mean_evidence} std={r.std_evidence} "
              f"auroc={r.evidence_auroc} -> {r.verdict}")
    print(f"  结论: {edl_summ['recommendation']}")

    # C1 conformal（用合成均匀概率演示覆盖率保证）
    rng = np.random.default_rng(9)
    n_hist = 300
    red_hist = [rng.dirichlet(np.ones(33) * 5) for _ in range(n_hist)]
    blue_hist = [rng.dirichlet(np.ones(16) * 5) for _ in range(n_hist)]
    red_draws = [rng.choice(33, 6, replace=False) + 1 for _ in range(n_hist)]
    blue_draws = [rng.integers(1, 17) for _ in range(n_hist)]
    cs = build_from_history(red_hist, blue_hist, red_draws, blue_draws, alpha=0.90)
    red_cov = np.mean([1 if all(dj in cs["red"].predict_set(red_hist[i]) for dj in red_draws[i])
                       else 0 for i in range(n_hist)])
    blue_cov = np.mean([1 if blue_draws[i] in cs["blue"].predict_set(blue_hist[i])
                        else 0 for i in range(n_hist)])
    print("\n[C1 Conformal 集合覆盖率]（合成均匀概率验证保证成立）：")
    print(f"  红球经验覆盖率={red_cov:.3f} 蓝球经验覆盖率={blue_cov:.3f} (目标 0.90)")


if __name__ == "__main__":
    run_real_evidence()
    print("\n[done] A/C 探针套件全部跑通")
