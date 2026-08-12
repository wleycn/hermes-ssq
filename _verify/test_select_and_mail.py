#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""select_numbers.py / send_ssq_picks.py 的纯逻辑验证(不依赖真实 PG/网络)。

运行: .venv/bin/python -m pytest _verify/test_select_and_mail.py -q
"""
from __future__ import annotations
import importlib.util
from pathlib import Path
import numpy as np
import pytest

ROOT = Path("/home/hermes/workspace/python/SSQ")


def _load(name, path):
    spec = importlib.util.spec_from_file_location(name, str(path))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


@pytest.fixture
def SN():
    return _load("select_numbers", ROOT / "select_numbers.py")


@pytest.fixture
def SSP():
    return _load("send_ssq_picks", ROOT / "send_ssq_picks.py")


@pytest.fixture
def mock_probs():
    red = np.random.RandomState(0).rand(33)
    red[[22, 15, 10, 4, 7]] = [0.9, 0.8, 0.7, 0.65, 0.6]
    blue = np.random.RandomState(1).rand(16)
    blue[5] = 0.9
    return red, blue


def test_generate_shape_and_constraints(SN, mock_probs):
    red, blue = mock_probs
    groups = SN.generate(red, blue, groups=5, seed=42)
    assert len(groups) == 5
    for g in groups:
        assert len(set(g["red"])) == 6, "红球重复"
        assert all(1 <= n <= 33 for n in g["red"])
        odds = sum(1 for x in g["red"] if x % 2 == 1)
        big = sum(1 for x in g["red"] if x > 16)
        assert odds in (2, 3, 4), f"奇偶比违例 {g}"
        assert big in (2, 3, 4), f"大小比违例 {g}"
        assert 1 <= g["blue"] <= 16
        assert isinstance(g["hot_reds"], list)


def test_generate_seeded_reproducible(SN, mock_probs):
    red, blue = mock_probs
    a = SN.generate(red, blue, groups=5, seed=7)
    b = SN.generate(red, blue, groups=5, seed=7)
    assert a == b, "相同 seed 应可复现"


def test_build_body_contains_picks_and_logic(SSP, mock_probs):
    red, blue = mock_probs
    body = SSP.build_body(red, blue, groups=5, run_at="2026-08-12 23:00:00", models=["rf", "lgbm", "cnn_math"])
    assert "双色球 5 组候选号码" in body
    assert "【选取逻辑】" in body
    assert "免责声明" in body
    # 5 组都应出现
    assert body.count("第") >= 5
    # 模型来源列出实际传入的模型
    for m in ["rf", "lgbm", "cnn_math"]:
        assert m in body


def test_send_email_dry_run(SSP, mock_probs, capsys):
    red, blue = mock_probs
    body = SSP.build_body(red, blue, groups=5, run_at="x", models=["rf", "lgbm", "cnn_math"])
    ok = SSP.send_email("主题", body, "wleycn@163.com", dry_run=True)
    assert ok is True
    out = capsys.readouterr().out
    assert "DRY-RUN" in out and "wleycn@163.com" in out
