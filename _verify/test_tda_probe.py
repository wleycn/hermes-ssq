#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""TDA 持久同调探针单测 (研究简报 2026-08-20 [2], 2026-08-22 补落地)。"""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import numpy as np
import pytest

from ml.probes.tda_probe import _embed, run_tda_probe

ripser = pytest.importorskip("ripser")


def test_embed_shape():
    """延迟嵌入形状: (n-(dim-1)*tau, dim)。"""
    rng = np.random.default_rng(1)
    x = rng.standard_normal(200)
    pc = _embed(x, dim=4, tau=1)
    assert pc.shape == (197, 4)


def test_tda_random_verdict():
    """白噪声: H1 最大寿命与 surrogate 无显著差异 → RANDOM。"""
    rng = np.random.default_rng(2)
    x = rng.standard_normal(500)
    res = run_tda_probe(x, n_surrogates=10)
    assert all(r.verdict == "RANDOM" for r in res), f"白噪声应 RANDOM: {res}"


def test_tda_detects_walk():
    """随机游走(强结构): H1 最大寿命显著高于 surrogate → NONRANDOM。"""
    rng = np.random.default_rng(3)
    x = np.cumsum(rng.standard_normal(400))
    res = run_tda_probe(x, n_surrogates=10)
    assert any(r.verdict == "NONRANDOM" for r in res), f"随机游走应 NONRANDOM: {res}"


def test_tda_invalid_short():
    """过短序列返回 INVALID。"""
    rng = np.random.default_rng(4)
    x = rng.standard_normal(50)
    res = run_tda_probe(x)
    assert res[0].verdict == "INVALID"


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
