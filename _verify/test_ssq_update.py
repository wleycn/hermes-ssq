#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""双色球更新脚本的可复现验证 (pytest)。

本文件刻意放在 SSQ 包之外（_verify/ 目录），避免 pytest 收集时触发
ml/data/__init__.py 的 numpy 依赖。

运行:  ~/.hermes/venv/bin/python -m pytest _verify/test_ssq_update.py -q
"""
from __future__ import annotations

import importlib.util
import shutil
from pathlib import Path

import pytest

SRC = Path("/home/hermes/workspace/python/SSQ/ml/data")
REAL_CSV = SRC / "1.csv"


def _load(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, str(path))
    assert spec is not None, f"无法加载 {path}"
    mod = importlib.util.module_from_spec(spec)
    import sys
    old = sys.argv
    sys.argv = ["x"]
    try:
        spec.loader.exec_module(mod)  # type: ignore[union-attr]
    finally:
        sys.argv = old
    return mod


@pytest.fixture
def append_mod():
    return _load("append_ssq", SRC / "append_ssq.py")


@pytest.fixture
def update_mod():
    return _load("update_ssq", SRC / "update_ssq.py")


@pytest.fixture
def tmp_csv(append_mod, tmp_path):
    dst = tmp_path / "1.csv"
    shutil.copy(REAL_CSV, dst)
    append_mod.CSV_PATH = dst
    return dst


def test_append_new_and_idempotent(append_mod, tmp_csv):
    rec = {"dNum": 2026093, "yNum": 2026, "mNum": 8, "dDate": "2026-08-13",
           "Red1": 1, "Red2": 2, "Red3": 3, "Red4": 4, "Red5": 5, "Red6": 6, "Blue1": 7}
    added, skipped = append_mod.append_records([rec])
    assert added == ["2026093"], added
    assert skipped == [], skipped

    added2, skipped2 = append_mod.append_records([rec])
    assert added2 == [], added2
    assert skipped2 == ["2026093"], skipped2


def test_append_preserves_crlf_and_format(append_mod, tmp_csv):
    rec = {"dNum": 2026093, "yNum": 2026, "mNum": 8, "dDate": "2026-08-13",
           "Red1": 1, "Red2": 2, "Red3": 3, "Red4": 4, "Red5": 5, "Red6": 6, "Blue1": 7}
    append_mod.append_records([rec])
    raw = tmp_csv.read_bytes()
    assert b"\r\n" in raw, "必须保留 CRLF 行尾"
    lines = raw.split(b"\r\n")
    last = lines[-2].decode()
    assert last.startswith("2026093,"), last
    parts = last.split(",")
    assert len(parts) == 11
    assert all(len(p) == 2 for p in parts[4:]), parts


def test_fetch_latest_real(update_mod):
    res = update_mod.fetch_latest()
    assert res is not None, "fetch_latest 应返回最新一期"
    assert res["dNum"] == 2026092, res
    assert res["reds"] == [9, 11, 12, 25, 30, 33], res
    assert res["blue"] == 11, res
    assert res["dDate"] == "2026-08-11", res
    assert res["mNum"] == 8, res


def test_cwl_official_source(update_mod):
    """中彩网官方接口应作为权威主源独立可用。"""
    res = update_mod.parse_cwl_latest()
    assert res is not None, "中彩网接口应返回数据"
    assert res["dNum"] == 2026092, res
    assert res["reds"] == [9, 11, 12, 25, 30, 33], res
    assert res["blue"] == 11, res
    assert res["dDate"] == "2026-08-11", res
