#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""RMT 探针单测 (研究简报 2026-08-20 [3], qwen 修正后 33号码×窗口规格)。"""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import numpy as np
import pytest

from ml.probes.rmt_probe import _mp_bounds, build_matrix, rmt_spectrum, run_rmt_probe


def test_mp_bounds_known_values():
    """MP 支集: q=1 时 [0,4]; q=4 时 [0.25, 2.25] 附近。"""
    lo, hi = _mp_bounds(1.0)
    assert abs(lo - 0.0) < 1e-6 and abs(hi - 4.0) < 1e-6
    lo4, hi4 = _mp_bounds(4.0)
    assert abs(lo4 - (1 - 2) ** 2) < 1e-6 and abs(hi4 - (1 + 2) ** 2) < 1e-6


def test_build_matrix_shape_and_values():
    """矩阵形状 N×T, 每期恰好 6 个 1。"""
    rng = np.random.default_rng(1)
    draws = [sorted(rng.choice(33, 6, replace=False) + 1) for _ in range(300)]
    X = build_matrix(draws, n_numbers=33, window=200)
    assert X.shape == (33, 200)
    assert np.all(X.sum(axis=0) == 6)          # 每期 6 红
    assert np.all(np.isin(X, (0, 1)))


def test_rmt_spectrum_random_draws_no_spikes():
    """纯随机开奖: max_ratio 不异常偏高(相对其 surrogate 分布判 RANDOM)。"""
    rng = np.random.default_rng(2)
    draws = [sorted(rng.choice(33, 6, replace=False) + 1) for _ in range(400)]
    X = build_matrix(draws, n_numbers=33, window=200)
    s = rmt_spectrum(X)
    # 小维度(N=33)有限样本下 max_eig 约 1.5~2.5, 远小于渐近 λ_+(≈12);
    # 因此不用绝对尖峰判据, 以 surrogate 相对显著性为准(见 run_rmt_probe)。
    assert 0.05 < s["max_ratio"] < 0.6, f"随机数据 max_ratio={s['max_ratio']} 异常"


def test_rmt_spectrum_detect_synthetic_coupling():
    """合成强耦合(6号共进退)应被 surrogate 判据检测为 NONRANDOM。"""
    rng = np.random.default_rng(3)
    draws = []
    for _ in range(400):
        if rng.random() < 0.5:
            draws.append(list(range(1, 7)))                     # 6 号完全同步
        else:
            draws.append(sorted(rng.choice(np.arange(7, 34), 6, replace=False) + 1))
    res = run_rmt_probe(draws, window=200, n_surrogates=20)
    assert any(r.verdict == "NONRANDOM" for r in res), \
        f"强耦合应被检出 NONRANDOM: {res}"


def test_run_probe_verdict_random_on_random_draws():
    """真实运行探针: 随机开奖两判据均 RANDOM。"""
    rng = np.random.default_rng(4)
    draws = [sorted(rng.choice(33, 6, replace=False) + 1) for _ in range(300)]
    res = run_rmt_probe(draws, window=150, n_surrogates=10)
    assert len(res) == 2
    assert all(r.verdict == "RANDOM" for r in res), f"随机开奖不应报 NONRANDOM: {res}"


def test_invalid_too_few_draws():
    with pytest.raises(ValueError):
        build_matrix([[1, 2, 3, 4, 5, 6]], window=200)


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
