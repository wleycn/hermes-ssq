#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""batch_predict_pg.py 的验证：聚焦 run_one 的概率提取与维度保护。

通过 monkeypatch 注入假模型输出, 避免触发真实训练(慢/需GPU)。
运行: .venv/bin/python -m pytest _verify/test_batch_predict_pg.py -q
"""
from __future__ import annotations
import importlib.util
from pathlib import Path
import numpy as np
import pytest

SRC = Path("/home/hermes/workspace/python/SSQ/batch_predict_pg.py")
ROOT = SRC.parent


def _load():
    import sys
    sys.path.insert(0, str(ROOT))
    spec = importlib.util.spec_from_file_location("batch_predict_pg", str(SRC))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


@pytest.fixture
def mod(monkeypatch):
    return _load()


def _fake_run_one(monkeypatch, payload):
    import ml.main as M
    monkeypatch.setattr(M, "run_train", lambda *a, **k: None)
    monkeypatch.setattr(M, "run_predict", lambda *a, **k: payload)


def test_full_prob_extraction_cnn(mod, monkeypatch):
    """cnn_math 形态: 全量概率应被正确提取为 33 红 / 16 蓝。"""
    payload = {"all_red_probs": [0.0] * 33, "all_blue_probs": [0.0] * 16}
    payload["all_red_probs"][22] = 0.9
    payload["all_blue_probs"][5] = 0.8
    _fake_run_one(monkeypatch, payload)
    out = mod.run_one("cnn_math", None)
    assert out is not None
    reds, blues = out
    assert reds.shape == (33,) and blues.shape == (16,)
    assert reds[22] == 0.9 and blues[5] == 0.8


def test_reject_cnn_missing_keys(mod, monkeypatch):
    """cnn_math 缺少全量概率键应被拒绝。"""
    _fake_run_one(monkeypatch, {"top_numbers": [1, 2, 3]})
    assert mod.run_one("cnn_math", None) is None


def test_reject_bad_dim(mod, monkeypatch):
    """红球维度不足33时, reds 应为 None(部分输出合法, 而非整体拒绝)。"""
    _fake_run_one(monkeypatch, {"all_red_probs": [0.1] * 10, "all_blue_probs": [0.1] * 16})
    reds, blues = mod.run_one("cnn_math", None)
    assert reds is None
    assert blues is not None and blues.shape == (16,)


def test_reject_exception(mod, monkeypatch):
    """模型抛异常应被捕获并返回 None(不崩溃)。"""
    import ml.main as M
    def _boom(*a, **k):
        raise RuntimeError("boom")
    monkeypatch.setattr(M, "run_train", _boom)
    assert mod.run_one("cnn_math", None) is None


def test_rf_aggregation(mod, monkeypatch):
    """rf/lgbm: batch_process 返回的逐位置结果应聚合成 33红/16蓝。"""
    import ml.main as M
    # 模拟 batch_process 返回 {rf_Red1: {all_numbers,all_probs}, rf_Blue1: {...}}
    def fake_batch(mt, df, cols, retrain):
        return {
            "rf_Red1": {"all_numbers": [1, 2, 3], "all_probs": [0.5, 0.3, 0.2]},
            "rf_Blue1": {"all_numbers": [11, 8, 1], "all_probs": [0.4, 0.3, 0.3]},
        }
    monkeypatch.setattr(M, "batch_process", fake_batch)
    out = mod.run_one("rf", None)
    assert out is not None
    reds, blues = out
    assert reds.shape == (33,) and blues.shape == (16,)
    # 红球 1/2/3 各累加对应概率; 蓝球 11/8/1 累加
    assert abs(reds[0] - 0.5) < 1e-9
    assert abs(reds[1] - 0.3) < 1e-9
    assert abs(reds[2] - 0.2) < 1e-9
    assert abs(blues[10] - 0.4) < 1e-9  # 蓝球11 -> idx10
    assert abs(blues[7] - 0.3) < 1e-9   # 蓝球8 -> idx7
    assert abs(blues[0] - 0.3) < 1e-9   # 蓝球1 -> idx0
