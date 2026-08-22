#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""James-Stein 收缩模块单测 (研究简报 2026-08-22 [1])。"""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import numpy as np
import pytest

from ml.shrinkage import james_stein_shrink, shrink_red_blue


def test_shrink_normalizes():
    """收缩后概率和为 1。"""
    rng = np.random.default_rng(1)
    p = rng.dirichlet(np.full(33, 0.9))
    js = james_stein_shrink(p)
    assert abs(js.sum() - 1.0) < 1e-9
    assert js.shape == p.shape


def test_shrink_pulls_toward_uniform():
    """偏离均匀的向量收缩后更接近均匀(最大概率变小)。"""
    p = np.zeros(33)
    p[0] = 0.9
    p[1:] = 0.1 / 32
    js = james_stein_shrink(p, sigma2=0.01)
    assert js.max() < 0.9, "收缩应降低峰值概率"


def test_shrink_alpha_zero_identity():
    """alpha=0 时不收缩(原样归一化)。"""
    p = np.zeros(16)
    p[0] = 0.8
    p[1:] = 0.2 / 15
    js = james_stein_shrink(p, alpha=0.0)
    assert abs(js[0] - 0.8) < 1e-9


def test_shrink_red_blue_shapes():
    """红33/蓝16 双通道收缩, 输出维度正确且归一化。"""
    rng = np.random.default_rng(2)
    r = rng.dirichlet(np.full(33, 1.0))
    b = rng.dirichlet(np.full(16, 1.0))
    rs, bs = shrink_red_blue(r, b)
    assert rs.shape == (33,) and bs.shape == (16,)
    assert abs(rs.sum() - 1) < 1e-9 and abs(bs.sum() - 1) < 1e-9


def test_shrink_mse_improvement_high_dim():
    """Stein 核心性质: 向均匀先验收缩在高维下降低 MSE(相对真值)。"""
    rng = np.random.default_rng(3)
    truth = np.full(33, 1 / 33)  # 真值即均匀(i.i.d. 假设)
    # 噪声估计: 偏离均匀的部分
    noisy = truth + rng.normal(0, 0.02, 33)
    noisy = np.clip(noisy, 1e-6, None)
    noisy = noisy / noisy.sum()
    mse_raw = np.mean((noisy - truth) ** 2)
    js = james_stein_shrink(noisy, sigma2=0.02 ** 2)
    mse_js = np.mean((js - truth) ** 2)
    assert mse_js < mse_raw, f"Stein 应降低 MSE: raw={mse_raw:.6f} js={mse_js:.6f}"


def test_shrink_rejects_low_dim():
    """d<3 应拒绝(Stein 要求 d≥3)。"""
    with pytest.raises(ValueError):
        james_stein_shrink([0.5, 0.5])


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
