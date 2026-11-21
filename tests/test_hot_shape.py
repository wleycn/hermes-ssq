#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""冷热形状约束回归测试 (2026-11-21, Rocky 拍板进链路)。

背景 (q://lesson-20261121-ssq-hot-shape):
  实测开奖每期 6 红中热球(最近30期频数Top11)个数 1-3 覆盖约 85% 期数。
  形状约束 = 每注热球个数 ∈ [1,3], 对齐开奖分布, 不提升命中率(等价性验证:
  配对 t=-1.6 不显著, pass_rate +0.0055)。
  约束实现: wheel.ticket_hot_ok + greedy_cover(ticket_hot_set, ticket_hot_range)。

运行: cd /home/hermes/workspace/python/SSQ && .venv/bin/python -m pytest tests/test_hot_shape.py -q
"""
from __future__ import annotations
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np

from select_numbers import build_wheel_tickets
from wheel import ticket_hot_ok

HOT_SET = {1, 3, 5, 7, 9, 11, 13, 15, 17, 19, 21}


def _real_probs():
    """2026112 实证的 6 模型均值集成概率 (归一化, 蓝球 15 号最高 0.1046)。"""
    red = np.full(33, 0.0303)
    red[np.array([0, 2, 10, 11, 16, 28])] = 0.0618
    red /= red.sum()
    blue = np.array([
        0.063961, 0.053, 0.059297, 0.058451, 0.061829, 0.051012, 0.071558,
        0.055835, 0.052685, 0.062483, 0.065041, 0.053354, 0.052977, 0.052702,
        0.104596, 0.066892,
    ])
    return red, blue / blue.sum()


def _hot_count(reds) -> int:
    return sum(1 for x in reds if x in HOT_SET)


def test_ticket_hot_ok_basics():
    """ticket_hot_ok: None=不约束; 区间内通过; 区间外拒绝。"""
    assert ticket_hot_ok([1, 2, 4, 6, 8, 10], None) is True
    assert ticket_hot_ok([1, 3, 5, 2, 4, 6], HOT_SET) is True   # 3 热
    assert ticket_hot_ok([1, 3, 2, 4, 6, 8], HOT_SET) is True   # 2 热
    assert ticket_hot_ok([1, 2, 4, 6, 8, 10], HOT_SET) is True   # 1 热(下界含)
    assert ticket_hot_ok([2, 4, 6, 8, 10, 12], HOT_SET) is False  # 0 热
    assert ticket_hot_ok([1, 3, 5, 7, 2, 4], HOT_SET) is False   # 4 热
    assert ticket_hot_ok([2, 4, 6, 8, 10, 12], HOT_SET, min_hot=0) is True  # 放宽下界


def test_wheel_hot_shape_inside_range():
    """传入 hot_set: 30 注热球个数全部 ∈ [1,3] (形状约束核心不变量)。"""
    red, blue = _real_probs()
    res = build_wheel_tickets(red, blue, max_notes=30, seed=2026113,
                              hot_set=HOT_SET, hot_range=(1, 3))
    tickets = res["tickets"]
    assert len(tickets) == 30
    for t in tickets:
        assert 1 <= _hot_count(t["reds"]) <= 3
    hc = res["coverage"]["hot_constraint"]
    assert hc["hard_pass"] == hc["hard_total"] == 30
    assert hc["range"] == [1, 3]


def test_wheel_hot_shape_off_compat():
    """hot_set=None: 走原行为, 向后兼容(与蓝球池同款兼容契约)。"""
    red, blue = _real_probs()
    res_none = build_wheel_tickets(red, blue, max_notes=30, seed=2026113, hot_set=None)
    res_default = build_wheel_tickets(red, blue, max_notes=30, seed=2026113)
    assert [t["reds"] for t in res_none["tickets"]] == \
           [t["reds"] for t in res_default["tickets"]]
    assert res_default["coverage"]["hot_constraint"]["hard_pass"] == \
           res_default["coverage"]["hot_constraint"]["hard_total"]
