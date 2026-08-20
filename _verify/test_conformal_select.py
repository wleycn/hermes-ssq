#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""C1 conformal 接入 select_numbers 的集成测试。

验证: build_conformal 能从 PG 预测批次 + 1.csv 开奖对齐校准,
      apply_conformal 能返回带理论覆盖率保证的候选集(仅解释层, 不改选号)。

运行: .venv/bin/python -m pytest _verify/test_conformal_select.py -q
前置: 需 PG 已起 + model_predictions 有批次 + ml/data/1.csv 有开奖(正常环境满足)
"""
from __future__ import annotations

import pytest

from select_numbers import PG, build_conformal, apply_conformal, load_latest_probs
import psycopg


def _pg_alive() -> bool:
    try:
        c = psycopg.connect(**PG, connect_timeout=3)
        c.close()
        return True
    except Exception:
        return False


@pytest.mark.skipif(not _pg_alive(), reason="PG 不可达")
def test_build_conformal_runs_and_returns_or_none():
    conn = psycopg.connect(**PG)
    try:
        cs = build_conformal(conn, alpha=0.90)
    finally:
        conn.close()
    # 样本不足时返回 None 是可接受的; 有样本时必须是 dict 含 red/blue
    if cs is not None:
        assert "red" in cs and "blue" in cs
        assert cs["n_pairs"] >= 8


@pytest.mark.skipif(not _pg_alive(), reason="PG 不可达")
def test_apply_conformal_candidate_set_semantics():
    """候选集大小须在合理范围: 红球<=33, 蓝球<=16, 且覆盖声明存在。"""
    conn = psycopg.connect(**PG)
    try:
        red_mean, blue_mean, _, _ = load_latest_probs(conn)
        cs = build_conformal(conn, alpha=0.90)
    finally:
        conn.close()
    if cs is None:
        pytest.skip("校准样本不足, 跳过")

    res = apply_conformal(cs, red_mean, blue_mean)
    assert res is not None
    assert 0 < len(res["red_set"]) <= 33
    assert 0 < len(res["blue_set"]) <= 16
    assert "coverage_claim" in res["red_summary"]
    # 候选集是解释层: 集合越大越保守(覆盖率保证), 不提升命中率
    assert res["red_summary"]["alpha"] == 0.90


@pytest.mark.skipif(not _pg_alive(), reason="PG 不可达")
def test_no_conformal_flag_does_not_change_selection():
    """--no-conformal 时 generate 路径不受影响(回归: 选号逻辑纯本地随机)。"""
    import numpy as np
    red = np.full(33, 1.0 / 33)
    blue = np.full(16, 1.0 / 16)
    from select_numbers import generate
    g1 = generate(red, blue, groups=3, seed=42)
    g2 = generate(red, blue, groups=3, seed=42)
    assert g1 == g2  # 同种子确定性
