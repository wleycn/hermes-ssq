#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""batch_predict_pg.py 的 PG 写入集成测试(真实连接本地 PG, 验证 SQL 语法/约束)。

运行: .venv/bin/python -m pytest _verify/test_pg_integration.py -q
需: 本地 docker Postgres 在线 (127.0.0.1:5432 hermes/hermes123)
"""
from __future__ import annotations
import importlib.util
from pathlib import Path
import pytest

SRC = Path("/home/hermes/workspace/python/SSQ/batch_predict_pg.py")
ROOT = SRC.parent

PG = dict(host="127.0.0.1", port=5432, user="hermes", password="hermes123", dbname="hermes")


@pytest.fixture
def mod():
    import sys
    sys.path.insert(0, str(ROOT))
    spec = importlib.util.spec_from_file_location("batch_predict_pg", str(SRC))
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def test_pg_insert_roundtrip(mod):
    """验证 ensure_schema + ON CONFLICT 写入 + 读回 真实可用。"""
    import sys
    sys.path.insert(0, str(ROOT))
    import importlib.util as _u
    spec_sn = _u.spec_from_file_location("select_numbers", str(ROOT / "select_numbers.py"))
    sn = _u.module_from_spec(spec_sn); spec_sn.loader.exec_module(sn)

    conn = None
    try:
        conn = __import__("psycopg").connect(**PG)
        mod.ensure_schema(conn)
        from datetime import datetime
        run_at = datetime.now()
        with conn.cursor() as cur:
            for i in range(1, 34):
                cur.execute(
                    f"INSERT INTO {mod.SCHEMA}.model_predictions(run_at,model,ball_type,num,prob) "
                    "VALUES (%s,%s,'red',%s,%s) "
                    "ON CONFLICT (run_at,model,ball_type,num) DO UPDATE SET prob=EXCLUDED.prob;",
                    (run_at, "test_rf", i, 1.0 / 33))
            for i in range(1, 17):
                cur.execute(
                    f"INSERT INTO {mod.SCHEMA}.model_predictions(run_at,model,ball_type,num,prob) "
                    "VALUES (%s,%s,'blue',%s,%s) "
                    "ON CONFLICT (run_at,model,ball_type,num) DO UPDATE SET prob=EXCLUDED.prob;",
                    (run_at, "test_rf", i, 1.0 / 16))
        conn.commit()
        red, blue, _, models = sn.load_latest_probs(conn)
        assert red.shape == (33,) and blue.shape == (16,)
        assert abs(red.sum() - 1.0) < 1e-6 and abs(blue.sum() - 1.0) < 1e-6
        assert "test_rf" in models
    finally:
        if conn:
            try:
                with conn.cursor() as cur:
                    cur.execute(f"DELETE FROM {mod.SCHEMA}.model_predictions WHERE model=%s", ("test_rf",))
                conn.commit()
            except Exception:
                pass
            conn.close()
