#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""验证 run_one 支持部分输出模型(lstm_blue仅蓝 / lstm_reds仅红), 及 select_numbers 均值逻辑。

运行: .venv/bin/python -m pytest _verify/test_partial_models.py -q
"""
from __future__ import annotations
import importlib.util
import sys
from pathlib import Path
import numpy as np
import pytest

SSQ = Path("/home/hermes/workspace/python/SSQ")


def _load(name, path):
    sys.path.insert(0, str(SSQ))
    spec = importlib.util.spec_from_file_location(name, str(path))
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


@pytest.fixture
def BPP():
    return _load("batch_predict_pg", SSQ / "batch_predict_pg.py")


@pytest.fixture
def SN():
    return _load("select_numbers", SSQ / "select_numbers.py")


def test_run_one_partial_blue(BPP, monkeypatch):
    """lstm_blue 仅输出蓝球, reds 应为 None。"""
    import ml.main as M
    monkeypatch.setattr(M, "run_train", lambda *a, **k: None)
    # lstm_blue 只返回 all_blue_probs
    monkeypatch.setattr(M, "run_predict", lambda *a, **k: {"all_blue_probs": [0.1] * 16})
    reds, blues = BPP.run_one("lstm_blue", None)
    assert reds is None
    assert blues is not None and blues.shape == (16,)


def test_run_one_partial_red(BPP, monkeypatch):
    """lstm_reds 仅输出红球, blues 应为 None。"""
    import ml.main as M
    monkeypatch.setattr(M, "run_train", lambda *a, **k: None)
    monkeypatch.setattr(M, "run_predict", lambda *a, **k: {"all_red_probs": [0.1] * 33})
    reds, blues = BPP.run_one("lstm_reds", None)
    assert reds is not None and reds.shape == (33,)
    assert blues is None


def test_run_one_both(BPP, monkeypatch):
    """cnn_math 红蓝全量。"""
    import ml.main as M
    monkeypatch.setattr(M, "run_train", lambda *a, **k: None)
    monkeypatch.setattr(M, "run_predict",
                        lambda *a, **k: {"all_red_probs": [0.1] * 33, "all_blue_probs": [0.1] * 16})
    reds, blues = BPP.run_one("cnn_math", None)
    assert reds.shape == (33,) and blues.shape == (16,)


def test_load_latest_probs_partial_mean(BPP, SN):
    """部分模型只贡献一侧时, 均值应按有数据侧计算(不引入 None)。"""
    import psycopg
    from datetime import datetime
    conn = psycopg.connect(host="127.0.0.1", port=5432, user="hermes",
                           password="hermes123", dbname="hermes")
    run_at = None
    try:
        BPP.ensure_schema(conn)  # ensure_schema 在 batch_predict_pg 中
        run_at = datetime.now()
        with conn.cursor() as cur:
            # lstm_blue 仅蓝, rf 红蓝全量
            for i in range(1, 34):
                cur.execute(
                    f"INSERT INTO {SN.SCHEMA}.model_predictions(run_at,model,ball_type,num,prob) "
                    "VALUES (%s,'rf','red',%s,%s) ON CONFLICT (run_at,model,ball_type,num) DO UPDATE SET prob=EXCLUDED.prob;",
                    (run_at, i, 1.0 / 33))
            for i in range(1, 17):
                cur.execute(
                    f"INSERT INTO {SN.SCHEMA}.model_predictions(run_at,model,ball_type,num,prob) "
                    "VALUES (%s,'rf','blue',%s,%s) ON CONFLICT (run_at,model,ball_type,num) DO UPDATE SET prob=EXCLUDED.prob;",
                    (run_at, i, 1.0 / 16))
                cur.execute(
                    f"INSERT INTO {SN.SCHEMA}.model_predictions(run_at,model,ball_type,num,prob) "
                    "VALUES (%s,'lstm_blue','blue',%s,%s) ON CONFLICT (run_at,model,ball_type,num) DO UPDATE SET prob=EXCLUDED.prob;",
                    (run_at, i, 2.0 / 16))  # 蓝球第二模型给更高概率
        conn.commit()
        red_mean, blue_mean, _, models = SN.load_latest_probs(conn)
        assert red_mean.shape == (33,) and blue_mean.shape == (16,)
        # 蓝球应综合 rf(1/16) 与 lstm_blue(2/16) -> 均值 1.5/16
        assert abs(blue_mean[0] - 1.5 / 16) < 1e-9
        # 红球仅 rf 贡献 -> 1/33
        assert abs(red_mean[0] - 1.0 / 33) < 1e-9
    finally:
        with conn.cursor() as cur:
            cur.execute(f"DELETE FROM {SN.SCHEMA}.model_predictions WHERE run_at=%s", (run_at,))
        conn.commit()
        conn.close()
