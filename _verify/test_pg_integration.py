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
        run_at = datetime(2099, 1, 1)  # 未来时间, 确保 '每模型取最新' 只取到本测试数据
        with conn.cursor() as cur:
            for i in range(1, 34):
                cur.execute(
                    f"INSERT INTO {mod.SCHEMA}.model_predictions(run_at,model,ball_type,num,prob) "
                    "VALUES (%s,%s,'red',%s,%s) "
                    "ON CONFLICT (run_at,model,ball_type,num) DO UPDATE SET prob=EXCLUDED.prob;",
                    (run_at, "t_rf", i, 1.0 / 33))
            for i in range(1, 17):
                cur.execute(
                    f"INSERT INTO {mod.SCHEMA}.model_predictions(run_at,model,ball_type,num,prob) "
                    "VALUES (%s,%s,'blue',%s,%s) "
                    "ON CONFLICT (run_at,model,ball_type,num) DO UPDATE SET prob=EXCLUDED.prob;",
                    (run_at, "t_rf", i, 1.0 / 16))
        conn.commit()
        # 注意: load_latest_probs 取 '每模型最新', 真实库已有 6 模型会混入。
        # 为隔离, 事务内先清空 model_predictions(仅预测结果, 可重建), 再插测试数据, 断言后 rollback。
        with conn.cursor() as cur:
            cur.execute(f"DELETE FROM {mod.SCHEMA}.model_predictions")
        with conn.cursor() as cur:
            for i in range(1, 34):
                cur.execute(
                    f"INSERT INTO {mod.SCHEMA}.model_predictions(run_at,model,ball_type,num,prob) "
                    "VALUES (%s,%s,'red',%s,%s) "
                    "ON CONFLICT (run_at,model,ball_type,num) DO UPDATE SET prob=EXCLUDED.prob;",
                    (run_at, "t_rf", i, 1.0 / 33))
            for i in range(1, 17):
                cur.execute(
                    f"INSERT INTO {mod.SCHEMA}.model_predictions(run_at,model,ball_type,num,prob) "
                    "VALUES (%s,%s,'blue',%s,%s) "
                    "ON CONFLICT (run_at,model,ball_type,num) DO UPDATE SET prob=EXCLUDED.prob;",
                    (run_at, "t_rf", i, 1.0 / 16))
        # 不 commit, 同连接可读到未提交数据(事务隔离), 避免污染真实 6 模型
        red, blue, _, models = sn.load_latest_probs(conn)
        assert red.shape == (33,) and blue.shape == (16,)
        # t_rf 是事务内唯一模型, 概率和应为 1.0
        assert abs(red.sum() - 1.0) < 1e-6 and abs(blue.sum() - 1.0) < 1e-6
        assert "t_rf" in models
    finally:
        if conn:
            try:
                conn.rollback()  # 回滚, 不污染真实 model_predictions
            except Exception:
                pass
            conn.close()
