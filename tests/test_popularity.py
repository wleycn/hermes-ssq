#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""ml/popularity.py 组合流行度规则评分与加权采样单测。

运行: cd /home/hermes/workspace/python/SSQ && .venv/bin/python -m pytest tests/test_popularity.py -q
"""
from __future__ import annotations
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np
import pytest

from ml.popularity import combo_popularity, popularity_penalty, sample_with_popularity

# 热门组合: 全奇(规则4命中) + 等差低起点(规则2) + 同尾 1/11(规则5) + 吉号 9(规则6)
POPULAR = [1, 3, 5, 7, 9, 11]
# 均衡组合: 奇偶均衡(3:3)、无连号、无等差、无同尾
BALANCED = [2, 4, 11, 13, 17, 22]

# 只保留 lucky_taboo 规则的权重(隔离单规则)
_LUCKY_ONLY = {
    "consecutive_pairs": 0.0, "arithmetic_progressions": 0.0,
    "birthday_ratio": 0.0, "all_same_parity": 0.0,
    "same_tail_pairs": 0.0, "lucky_taboo": 1.0,
}


# ① 全奇/连号/同尾组合流行度 > 均衡组合
def test_popular_combo_scores_higher():
    p_pop = combo_popularity(POPULAR)
    p_bal = combo_popularity(BALANCED)
    assert p_pop > p_bal
    # 热门组合的各命中规则本身也应高于均衡组合
    assert p_pop > 0.5


# ② 含 6/8/9 流行度 > 含 4 (lucky_taboo 规则隔离对比)
def test_lucky_gt_taboo():
    lucky = combo_popularity([6, 8, 9, 13, 17, 22], weights=_LUCKY_ONLY)
    taboo = combo_popularity([4, 14, 24, 13, 17, 22], weights=_LUCKY_ONLY)
    assert lucky > taboo
    assert lucky == pytest.approx(0.5)   # (3-0)/6
    assert taboo == pytest.approx(0.0)   # (0-1)/6 → 截断 0


# ③ combo_popularity ∈ [0,1]
def test_combo_popularity_in_unit_interval():
    rng = np.random.default_rng(0)
    for _ in range(20):
        reds = sorted(rng.choice(33, size=6, replace=False).tolist())
        p = combo_popularity(reds)
        assert 0.0 <= p <= 1.0
        assert 0.0 <= combo_popularity(reds, weights=_LUCKY_ONLY) <= 1.0


# ④ popularity_penalty 随 lambda 单调(λ大→penalty小)
def test_penalty_monotonic_in_lambda():
    assert combo_popularity(POPULAR) > 0.0
    assert popularity_penalty(POPULAR, lambda_=0.0) == pytest.approx(1.0)
    prev = 1.0
    for lam in (0.0, 0.3, 0.6, 0.9):
        pen = popularity_penalty(POPULAR, lambda_=lam)
        assert pen <= prev + 1e-9, f"λ={lam} 时应单调不增"
        prev = pen
    assert popularity_penalty(POPULAR, lambda_=0.9) < popularity_penalty(POPULAR, lambda_=0.3)


def test_penalty_in_unit_interval():
    rng = np.random.default_rng(1)
    for _ in range(10):
        reds = sorted(rng.choice(33, size=6, replace=False).tolist())
        pen = popularity_penalty(reds, lambda_=1.5)  # λ>1 也应被 clip 到 [0,1]
        assert 0.0 <= pen <= 1.0


# ⑤ sample_with_popularity 输出 6 个互异号码
def test_sample_with_popularity_output():
    rng = np.random.default_rng(42)
    red_prob = np.random.RandomState(0).rand(33)
    red_prob[[22, 15, 10, 4, 7]] = [0.9, 0.8, 0.7, 0.65, 0.6]
    for _ in range(5):
        reds = sample_with_popularity(red_prob, rng, n_candidates=50)
        assert len(set(reds)) == 6, "号码重复"
        assert all(1 <= n <= 33 for n in reds)
        assert reds == sorted(reds)
