#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""数据层改造验证（T6）：draw_history 导入 / data_date 写入 / 30天清理 dry-run。

运行: .venv/bin/python -m pytest _verify/test_data_layer.py -q
依赖真实 PG（127.0.0.1:5432 hermes/ssq）；不会删除任何业务数据。
"""
from __future__ import annotations

import importlib.util
import sys
from datetime import datetime, date, timedelta
from pathlib import Path

import psycopg
import pytest

SSQ = Path("/home/hermes/workspace/python/SSQ")
CSV = SSQ / "ml/data/1.csv"
EXPECTED_ROWS = 3488


def _load(name: str, path: Path):
    sys.path.insert(0, str(SSQ))
    spec = importlib.util.spec_from_file_location(name, str(path))
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


@pytest.fixture
def conn():
    c = psycopg.connect(host="127.0.0.1", port=5432, user="hermes",
                        password="hermes123", dbname="hermes")
    yield c
    c.close()


@pytest.fixture
def schema(conn):
    pg = _load("pg_schema", SSQ / "pg_schema.py")
    pg.ensure_draw_history_schema(conn)
    pg.ensure_model_predictions_data_date(conn)
    return pg


# --------------------------------------------------------------------------- #
# (a) import_draw_history 行数=3488 且首末行正确
# --------------------------------------------------------------------------- #
def test_import_draw_history_count(conn, schema):
    n = schema.import_draw_history(CSV, conn)
    assert n == EXPECTED_ROWS
    with conn.cursor() as cur:
        cur.execute("SELECT COUNT(*) FROM ssq.draw_history;")
        assert cur.fetchone()[0] == EXPECTED_ROWS


def test_import_draw_history_edges(conn, schema):
    schema.import_draw_history(CSV, conn)  # 幂等重导
    with conn.cursor() as cur:
        cur.execute(
            "SELECT dNum, dDate FROM ssq.draw_history "
            "ORDER BY dDate ASC, dNum ASC LIMIT 1;"
        )
        first = cur.fetchone()
        cur.execute(
            "SELECT dNum, dDate FROM ssq.draw_history "
            "ORDER BY dDate DESC, dNum DESC LIMIT 1;"
        )
        last = cur.fetchone()
    assert first == ("2003001", date(2003, 2, 23))
    assert last == ("2026092", date(2026, 8, 11))


# --------------------------------------------------------------------------- #
# (b) data_date 写入非空且 == run_at.date()（北京日期）
# --------------------------------------------------------------------------- #
def test_data_date_non_null(conn, schema):
    with conn.cursor() as cur:
        cur.execute(
            "SELECT COUNT(*) FROM ssq.model_predictions WHERE data_date IS NULL;"
        )
        assert cur.fetchone()[0] == 0


def test_insert_explicit_data_date_matches_run_at(conn, schema):
    """模拟 batch_predict_pg.main() 的 INSERT：显式传 data_date=run_at.date()。"""
    run_at = datetime.now()
    data_date = run_at.date()
    with conn.cursor() as cur:
        cur.execute(
            "INSERT INTO ssq.model_predictions(run_at,model,ball_type,num,prob,data_date) "
            "VALUES (%s,'devtest_dl','red',1,0.5,%s)",
            (run_at, data_date),
        )
        conn.commit()
        cur.execute(
            "SELECT data_date FROM ssq.model_predictions "
            "WHERE model='devtest_dl' AND run_at=%s;",
            (run_at,),
        )
        assert cur.fetchone()[0] == data_date
    # 清理测试行
    with conn.cursor() as cur:
        cur.execute("DELETE FROM ssq.model_predictions WHERE model='devtest_dl';")
        conn.commit()


def test_existing_production_rows_data_date_is_beijing(conn, schema):
    """真实跑批写入的 245 行，data_date 应与各自 run_at 的北京日期一致。"""
    with conn.cursor() as cur:
        cur.execute(
            "SELECT COUNT(*) FROM ssq.model_predictions "
            "WHERE data_date IS DISTINCT FROM run_at::date;"
        )
        mismatched = cur.fetchone()[0]
        # 允许其它测试遗留的 DEFAULT 兜底行：data_date=北京今天 且与 run_at 不一致的行。
        # 不能再用 CURRENT_DATE(UTC) 识别——北京 08:00-24:00 时段 UTC 与北京同日，
        # 生产行(北京日期)会被误判为兜底行（0 != 245 缺陷根因）。
        cur.execute(
            "SELECT COUNT(*) FROM ssq.model_predictions "
            "WHERE data_date = (now() AT TIME ZONE 'Asia/Shanghai')::date "
            "  AND data_date IS DISTINCT FROM run_at::date;"
        )
        default_rows = cur.fetchone()[0]
    assert mismatched == default_rows


# --------------------------------------------------------------------------- #
# (c) cleanup --dry-run 计数为 0（当前数据都在 30 天内），且未删任何行
# --------------------------------------------------------------------------- #
def test_cleanup_dry_run_zero_and_noop(conn, schema):
    cp = _load("cleanup_predictions", SSQ / "cleanup_predictions.py")
    cutoff = datetime.now().date() - timedelta(days=30)

    with conn.cursor() as cur:
        before_mp = cur.execute("SELECT COUNT(*) FROM ssq.model_predictions").fetchone()[0]
        before_dh = cur.execute("SELECT COUNT(*) FROM ssq.draw_history").fetchone()[0]

    will_delete = cp.keep_recent_30d(conn, dry_run=True)

    with conn.cursor() as cur:
        after_mp = cur.execute("SELECT COUNT(*) FROM ssq.model_predictions").fetchone()[0]
        after_dh = cur.execute("SELECT COUNT(*) FROM ssq.draw_history").fetchone()[0]

    assert will_delete == 0
    assert after_mp == before_mp  # dry-run 未删任何行
    assert after_dh == before_dh  # draw_history 绝不被触碰
    assert cutoff == datetime.now().date() - timedelta(days=30)
