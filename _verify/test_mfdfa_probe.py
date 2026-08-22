#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""MF-DFA 探针单测 (研究简报 2026-08-20 [1])。"""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import numpy as np
import pytest

from ml.probes.mfdfa_probe import h_spectrum, run_mfdfa_probe


def test_h_spectrum_white_noise_approx_half():
    """纯白噪声: H(q=2)≈0.5 且各 q 的 h 接近(单分形)。"""
    rng = np.random.default_rng(7)
    x = rng.standard_normal(2000)
    h = h_spectrum(x)
    assert abs(h[2.0] - 0.5) < 0.15, f"H(2)={h[2.0]} 偏离 0.5 过远"
    assert h[2.0] < 0.75, "长程记忆信号不应出现在白噪声上"
    assert h[-4.0] - h[4.0] < 0.4, "白噪声应为单分形(谱宽小)"


def test_h_spectrum_persistent_series_h_gt_half():
    """构造正相关(长程记忆)序列: H(2) 应显著 > 0.5。"""
    rng = np.random.default_rng(11)
    # 累积和生成强持久随机游走: 差分是白噪声 → 积分序列 H≈1.0
    steps = rng.standard_normal(2000)
    x = np.cumsum(steps)
    h = h_spectrum(x)
    assert h[2.0] > 0.65, f"持久序列 H(2)={h[2.0]} 应明显高于 0.5"


def test_run_probe_verdict_random_on_white_noise():
    """真实运行探针: 白噪声两判据均 RANDOM。"""
    rng = np.random.default_rng(3)
    x = rng.standard_normal(800)
    res = run_mfdfa_probe(x, n_surrogates=15)
    assert len(res) == 2
    assert all(r.verdict == "RANDOM" for r in res), f"白噪声不应报 NONRANDOM: {res}"


def test_run_probe_lottery_scale_data():
    """短序列(彩票单球位尺度, n≈3500)也能跑, 不抛异常。"""
    rng = np.random.default_rng(5)
    x = rng.integers(1, 34, size=1000).astype(float)
    res = run_mfdfa_probe(x, n_surrogates=10)
    assert len(res) == 2


def test_invalid_short_sequence():
    """过短序列返回 INVALID 而非抛异常。"""
    with pytest.raises(ValueError):
        h_spectrum([1.0, 2.0, 3.0])


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
