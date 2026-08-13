#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""wheel.py 旋转矩阵贪心覆盖生成器单测。

运行: cd /home/hermes/workspace/python/SSQ && .venv/bin/python -m pytest tests/test_wheel.py -q
"""
from __future__ import annotations
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import pytest

from wheel import CoverResult, greedy_cover


def _pool(n: int):
    return list(range(1, n + 1))


# ① 参数校验 5 条各触发 ValueError
def test_validation_pool_too_small():
    with pytest.raises(ValueError, match="红球池大小必须"):
        greedy_cover([1, 2, 3, 4, 5])  # len=5 < k=6


def test_validation_number_out_of_range():
    with pytest.raises(ValueError, match="1-33"):
        greedy_cover([1, 2, 3, 4, 5, 34])  # 34 > 33
    with pytest.raises(ValueError, match="1-33"):
        greedy_cover([0, 1, 2, 3, 4, 5])   # 0 < 1


def test_validation_t_invalid():
    with pytest.raises(ValueError, match="t"):
        greedy_cover(_pool(12), t=6)  # 不满足 t < k
    with pytest.raises(ValueError, match="t"):
        greedy_cover(_pool(12), t=0)  # 不满足 t ≥ 1


def test_validation_max_notes():
    with pytest.raises(ValueError, match="max_notes"):
        greedy_cover(_pool(12), max_notes=5)  # 5 < k=6


def test_validation_restarts():
    with pytest.raises(ValueError, match="restarts"):
        greedy_cover(_pool(12), restarts=0)


def test_validation_pool_over_33():
    # 34 个号码超出双色球红球域(1-33), 必须被拒
    with pytest.raises(ValueError):
        greedy_cover(list(range(1, 35)))


# ② 每注 6 个互不重复升序号码且 ∈ pool
def test_tickets_well_formed():
    pool = _pool(15)
    res = greedy_cover(pool, max_notes=30, seed=0)
    assert res.tickets
    for t in res.tickets:
        assert len(t) == 6
        assert len(set(t)) == 6, "红球重复"
        assert t == sorted(t), "未升序"
        assert all(n in pool for n in t), "号码不在池内"


# ③ n_notes ≤ max_notes
def test_n_notes_bounded():
    for n in (10, 15):
        res = greedy_cover(_pool(15), max_notes=n, seed=0)
        assert res.n_notes <= n
        assert len(res.tickets) == res.n_notes


# ④ pool=15, max_notes=30 → pass_rate ≥ 0.99 (架构实测 100%)
def test_pool15_pass_rate():
    res = greedy_cover(_pool(15), max_notes=30, restarts=3, seed=0)
    assert res.pass_rate >= 0.99
    assert res.pass_rate_sampled is None  # N≤20 精确计算, 无抽样


# ⑤ pool=12 → pass_rate ≥ 0.99 (架构实测 100%)
def test_pool12_pass_rate():
    res = greedy_cover(_pool(12), max_notes=30, restarts=3, seed=0)
    assert res.pass_rate >= 0.99


# ⑥ pool=18 → 不崩溃、报告字段齐全、pass_rate ≥ 0.90 (防御性下界)
def test_pool18_smoke():
    res = greedy_cover(_pool(18), max_notes=30, restarts=3, seed=0)
    assert res.pass_rate >= 0.90
    assert isinstance(res, CoverResult)
    assert res.total_4subsets == 3060  # C(18,4)
    assert 0.0 <= res.four_subset_coverage <= 1.0
    assert 0 <= res.covered_4subsets <= res.total_4subsets
    assert isinstance(res.converged, bool)
    assert res.n_notes <= 30
    assert res.max_notes == 30


# ⑦ 同 seed 结果逐字节一致
def test_deterministic_same_seed():
    a = greedy_cover(_pool(15), max_notes=25, restarts=3, seed=7)
    b = greedy_cover(_pool(15), max_notes=25, restarts=3, seed=7)
    assert a == b
    assert a.tickets == b.tickets
    assert a.pass_rate == b.pass_rate
    assert a.covered_4subsets == b.covered_4subsets
    assert a.four_subset_coverage == b.four_subset_coverage
