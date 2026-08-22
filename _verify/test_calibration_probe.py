#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""校准探针单测 (研究简报 2026-08-22 [2], 落地 2026-08-22)。"""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import numpy as np
import pytest

from ml.probes.calibration_probe import (
    brier_score,
    ece,
    evaluate_calibration,
    isotonic_delta,
)


def test_brier_perfect_zero():
    p = np.zeros(33)
    p[4] = 1.0
    assert brier_score(p, 5) == 0.0


def test_brier_uniform_baseline():
    p = np.full(33, 1 / 33)
    b = brier_score(p, 5)
    assert 0.02 < b < 0.04, f"均匀 Brier={b} 应在基准附近"


def test_ece_uniform_output_small():
    """均匀输出(诚实): ECE 应接近 0。"""
    rng = np.random.default_rng(1)
    probs = np.stack([rng.dirichlet(np.full(33, 1.0)) for _ in range(300)])
    outs = rng.integers(1, 34, size=300)
    assert ece(probs, outs) < 0.02, "诚实输出 ECE 应接近 0"


def test_ece_detect_overconfidence():
    """尖峰高估: ECE 应明显大于诚实输出。"""
    rng = np.random.default_rng(2)
    n = 300
    probs = np.full((n, 33), 0.3 / 32)
    peaks = rng.integers(0, 33, size=n)
    probs[np.arange(n), peaks] = 0.7
    outs = rng.integers(1, 34, size=n)
    assert ece(probs, outs) > 0.03, "高估输出 ECE 应明显 > 0"


def test_isotonic_delta_reduces_brier_on_overconfidence():
    """高估输出: isotonic 校准应降低 Brier(delta<0)。"""
    rng = np.random.default_rng(2)
    n = 300
    probs = np.full((n, 33), 0.3 / 32)
    peaks = rng.integers(0, 33, size=n)
    probs[np.arange(n), peaks] = 0.7
    outs = rng.integers(1, 34, size=n)
    d = isotonic_delta(probs, outs)
    assert d < 0, f"高估时校准应降低 Brier, delta={d}"


def test_evaluate_calibration_returns_three_metrics():
    rng = np.random.default_rng(3)
    probs = np.stack([rng.dirichlet(np.full(16, 1.0)) for _ in range(100)])
    outs = rng.integers(1, 17, size=100)
    res = evaluate_calibration(probs, outs)
    assert [r.metric for r in res] == ["brier", "ece", "isotonic_delta"]
    assert all(np.isfinite(r.value) for r in res)


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
