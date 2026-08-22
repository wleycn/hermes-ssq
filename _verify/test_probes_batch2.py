#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""08-22 补落地探针单测: 可见图/LZ/RQA/MSE/Rényi/DCCA。"""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import numpy as np
import pytest

from ml.probes.visibility_probe import visibility_graph, run_visibility_probe
from ml.probes.lz_probe import lz76_complexity, run_lz_probe
from ml.probes.rqa_probe import run_rqa_probe
from ml.probes.mse_probe import mse_curve, run_mse_probe
from ml.probes.renyi_probe import renyi_entropy, run_renyi_probe
from ml.probes.dcca_probe import dcca_lambda, run_dcca_probe


# ---------- LZ76 核心 ----------
def test_lz76_constant_sequence():
    """全同序列复杂度≈1(高度可压缩)。"""
    assert lz76_complexity([0] * 10) <= 2


def test_lz76_random_approx_theory():
    """随机 0/1 复杂度≈n/log2(n)。"""
    rng = np.random.default_rng(1)
    c = lz76_complexity(list(rng.integers(0, 2, 200)))
    theory = 200 / np.log2(200)
    assert 0.5 * theory < c < 2.0 * theory


def test_lz_run_random_verdict():
    """随机整数序列: ratio≈1 → RANDOM。"""
    rng = np.random.default_rng(7)
    x = rng.integers(1, 34, size=300).astype(float)
    res = run_lz_probe(x, n_surrogates=15)
    assert res[0].verdict == "RANDOM"


# ---------- 可见图 ----------
def test_visibility_graph_symmetry():
    """邻接矩阵对称、无自环。"""
    rng = np.random.default_rng(2)
    x = rng.standard_normal(50)
    adj = visibility_graph(x)
    assert adj.shape == (50, 50)
    assert np.all(adj == adj.T)
    assert not np.any(np.diag(adj))


def test_visibility_random_verdict():
    """白噪声: λ 与 surrogate 分布无显著差异 → RANDOM。"""
    rng = np.random.default_rng(3)
    x = rng.standard_normal(400)
    res = run_visibility_probe(x, n_surrogates=15)
    assert all(r.verdict == "RANDOM" for r in res), f"白噪声应 RANDOM: {res}"


def test_visibility_detects_walk():
    """随机游走(强结构): 应判 NONRANDOM。"""
    rng = np.random.default_rng(4)
    x = np.cumsum(rng.standard_normal(300))
    res = run_visibility_probe(x, n_surrogates=15)
    assert any(r.verdict == "NONRANDOM" for r in res), f"随机游走应 NONRANDOM: {res}"


# ---------- RQA ----------
def test_rqa_random_verdict():
    """白噪声: DET/Lmax 与 surrogate 无显著差异 → RANDOM。"""
    rng = np.random.default_rng(5)
    x = rng.standard_normal(300)
    res = run_rqa_probe(x, n_surrogates=15)
    assert all(r.verdict == "RANDOM" for r in res), f"白噪声应 RANDOM: {res}"


def test_rqa_detects_walk():
    """随机游走: DET 显著高 → NONRANDOM。"""
    rng = np.random.default_rng(6)
    x = np.cumsum(rng.standard_normal(300))
    res = run_rqa_probe(x, n_surrogates=15)
    assert any(r.verdict == "NONRANDOM" for r in res), f"随机游走应 NONRANDOM: {res}"


# ---------- MSE ----------
def test_mse_curve_shapes():
    """MSE 曲线: 尺度 1..8 均有值。"""
    rng = np.random.default_rng(7)
    x = rng.standard_normal(500)
    curve = mse_curve(x, max_scale=8)
    assert 1 in curve and 8 in curve
    assert all(np.isfinite(v) for v in curve.values())


def test_mse_random_verdict():
    """白噪声: MSE 曲线不显著下降(decay<0.35) → RANDOM。"""
    rng = np.random.default_rng(8)
    x = rng.standard_normal(800)
    res = run_mse_probe(x)
    decay = [r for r in res if r.metric == "decay"][0]
    assert decay.verdict == "RANDOM", f"白噪声 MSE 不应显著衰减: {decay}"


# ---------- Rényi ----------
def test_renyi_shannon_limit():
    """q→1 退化为 Shannon 熵。"""
    p = np.array([0.5, 0.25, 0.25])
    assert abs(renyi_entropy(p, 1.0) - 1.0397) < 0.01


def test_renyi_uniform_flat_spectrum():
    """均匀分布: H4/H1≈1。"""
    rng = np.random.default_rng(9)
    x = rng.integers(1, 34, size=1000).astype(float)
    res = run_renyi_probe(x, n_classes=33)
    assert res[0].verdict == "RANDOM", f"均匀应 RANDOM: {res}"


# ---------- DCCA ----------
def test_dcca_self_degenerates_dfa():
    """同序列 DCCA λ≈0.5(退化为 DFA)。"""
    rng = np.random.default_rng(10)
    x = rng.standard_normal(2000)
    lam = dcca_lambda(x, x)
    assert abs(lam - 0.5) < 0.15, f"同序列 λ={lam} 应≈0.5"


def test_dcca_independent_random():
    """独立白噪声: λ 落在 [0.3,0.9] → RANDOM。"""
    rng = np.random.default_rng(11)
    a = rng.standard_normal(600)
    b = rng.standard_normal(600)
    res = run_dcca_probe(a, b)
    assert all(r.verdict == "RANDOM" for r in res), f"独立应 RANDOM: {res}"


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
