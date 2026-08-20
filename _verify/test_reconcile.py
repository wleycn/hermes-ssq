#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""reconcile_picks.py 纯逻辑验证(不依赖真实 PG/网络)。

运行: .venv/bin/python -m pytest _verify/test_reconcile.py -q
"""
from __future__ import annotations
import importlib.util
from pathlib import Path
import pytest

ROOT = Path("/home/hermes/workspace/python/SSQ")


def _load(name, path):
    spec = importlib.util.spec_from_file_location(name, str(path))
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


@pytest.fixture
def RC():
    return _load("reconcile_picks", ROOT / "reconcile_picks.py")


def _mk_pick(mode, gidx, reds, blue, popularity=0.5, seed=1):
    return {"mode": mode, "group_idx": gidx,
            "reds": reds, "blue": blue,
            "popularity": popularity, "seed": seed}


# ================= 1. analyze_hits 命中统计正确性 =================

def test_analyze_hits_red_blue_both_hit(RC):
    drawn_reds = [3, 8, 15, 21, 27, 30]
    drawn_blue = 5
    picks = [_mk_pick("top5", 1, [3, 8, 15, 22, 28, 31], 5)]
    a = RC.analyze_hits(drawn_reds, drawn_blue, picks)[0]
    assert a["red_count"] == 3
    assert a["hit_reds"] == [3, 8, 15]
    assert a["blue_hit"] is True
    assert a["prize"] == "五等奖"  # 3红+蓝


def test_analyze_hits_only_blue(RC):
    a = RC.analyze_hits([1, 2, 3, 4, 5, 6], 9, [_mk_pick("wheel", 1, [10, 11, 12, 13, 14, 15], 9)])[0]
    assert a["red_count"] == 0
    assert a["blue_hit"] is True
    assert a["prize"] == "六等奖"


def test_analyze_hits_no_hit(RC):
    a = RC.analyze_hits([1, 2, 3, 4, 5, 6], 9, [_mk_pick("top5", 1, [10, 11, 12, 13, 14, 15], 16)])[0]
    assert a["red_count"] == 0
    assert a["blue_hit"] is False
    assert a["prize"] == "未中奖"


def test_analyze_hits_six_reds_no_blue(RC):
    a = RC.analyze_hits([1, 2, 3, 4, 5, 6], 9, [_mk_pick("top5", 1, [1, 2, 3, 4, 5, 6], 10)])[0]
    assert a["red_count"] == 6
    assert a["blue_hit"] is False
    assert a["prize"] == "二等奖"


def test_analyze_hits_jackpot(RC):
    a = RC.analyze_hits([1, 2, 3, 4, 5, 6], 9, [_mk_pick("top5", 1, [1, 2, 3, 4, 5, 6], 9)])[0]
    assert a["red_count"] == 6
    assert a["blue_hit"] is True
    assert a["prize"] == "一等奖"


# ================= 2. prize_tier 奖级边界 =================

@pytest.mark.parametrize("r,b,expected", [
    (6, True, "一等奖"), (6, False, "二等奖"), (5, True, "三等奖"),
    (5, False, "四等奖"), (4, True, "四等奖"), (4, False, "五等奖"),
    (3, True, "五等奖"), (3, False, "未中奖"), (2, True, "六等奖"),
    (1, True, "六等奖"), (0, True, "六等奖"), (0, False, "未中奖"),
])
def test_prize_tier_boundaries(RC, r, b, expected):
    assert RC.prize_tier(r, b) == expected


# ================= 3. render_hits_html 输出结构 =================

def test_render_html_contains_both_tables(RC):
    picks = [_mk_pick("top5", i, [1, 2, 3, 4, 5, 6 + i], 9, popularity=0.7) for i in range(1, 6)]
    picks += [_mk_pick("wheel", i, [1, 2, 3, 4, 5, 6 + i], 9) for i in range(1, 31)]
    analyzed = RC.analyze_hits([1, 2, 3, 4, 5, 6], 9, picks)
    html = RC.render_hits_html("2026096", [1, 2, 3, 4, 5, 6], 9, analyzed)
    assert "A. 常规 5 组" in html
    assert "B. 旋转矩阵 wheel 30 注" in html
    assert html.count("<tr>") >= 5 + 30  # 5 组 + 30 注 + 2 表头
    assert "2026096" in html
    assert "✅" in html


def test_render_html_marks_hit_and_miss(RC):
    """命中号带 ✅ 且红色, 未中号灰色无标记。"""
    picks = [_mk_pick("top5", 1, [1, 7, 8, 9, 10, 11], 5)]
    analyzed = RC.analyze_hits([1, 2, 3, 4, 5, 6], 5, picks)
    html = RC.render_hits_html("P", [1, 2, 3, 4, 5, 6], 5, analyzed)
    assert "01✅" in html          # 命中红球带勾
    assert "color:#c00" in html    # 红色高亮
    assert "07" in html            # 未中号仍在
    assert "color:#999" in html    # 未中灰
    assert "05✅" in html and "color:#1e5aff" in html  # 蓝球命中蓝色


def test_render_html_no_picks_returns_none(RC):
    assert RC.build_reconcile_block.__doc__  # 存在对外入口
    # fetch 依赖 PG, 纯函数侧只验证: 空 analyzed 时渲染不含表
    html = RC.render_hits_html("P", [1, 2], 3, [])
    assert "A. 常规 5 组" not in html
    assert "B. 旋转矩阵" not in html
    assert "无推荐" not in html  # 占位文案由 update_ssq.py 负责
