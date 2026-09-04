#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""SSQ PostgreSQL 数据层辅助：开奖表 draw_history 建表/导入 + model_predictions.data_date 列迁移。

设计要点（依据 handoff_arch.json ADR-1..8 与实测修正）：
  - ssq.draw_history：1.csv 的全量镜像归档，11 列（dNum VARCHAR(8) 保留前导零），
    PK dNum，索引 ix_draw_history_ddate；永不清理（ADR-1）。
  - ssq.model_predictions：新增 data_date DATE，NOT NULL DEFAULT CURRENT_DATE（兜底），
    存量行用 run_at 经 Asia/Shanghai 时区转换回填为北京日期（修正 PG Etc/UTC 的 CURRENT_DATE 偏差）。
  - 所有“今天/日期”语义统一以 Python 侧 datetime.now()（服务器=Asia/Shanghai）为准，
    不依赖 PG 的 CURRENT_DATE（UTC，比北京晚 8h）。
"""
from __future__ import annotations

import sys
from datetime import datetime, date, timedelta
from pathlib import Path

import pandas as pd
import psycopg
from ml.pg_conn import pg_dict

SCHEMA = "ssq"
PG = pg_dict()  # 凭证从 ~/.hermes/.env 的 DATABASE_URL 读, 不硬编码

# 1.csv 表头 11 列顺序
DRAW_COLUMNS = ["dNum", "yNum", "mNum", "dDate", "Red1", "Red2", "Red3",
                "Red4", "Red5", "Red6", "Blue1"]
EXPECTED_ROWS = 3494  # 不含表头; 仅 __main__ 打印期望值, 非硬约束(数据增长会自动超过)


def get_conn() -> psycopg.Connection:
    """建立并返 PG 连接。"""
    return psycopg.connect(**PG)


# --------------------------------------------------------------------------- #
# T1: 开奖表建表
# --------------------------------------------------------------------------- #
def ensure_draw_history_schema(conn: psycopg.Connection) -> None:
    """CREATE TABLE IF NOT EXISTS ssq.draw_history（11 列 + PK dNum + 索引）。幂等。"""
    with conn.cursor() as cur:
        cur.execute(f"CREATE SCHEMA IF NOT EXISTS {SCHEMA};")
        cur.execute(f"""
        CREATE TABLE IF NOT EXISTS {SCHEMA}.draw_history (
            dNum   VARCHAR(8)  NOT NULL,
            yNum   SMALLINT   NOT NULL,
            mNum   SMALLINT   NOT NULL,
            dDate  DATE       NOT NULL,
            Red1   SMALLINT   NOT NULL CHECK (Red1 BETWEEN 1 AND 33),
            Red2   SMALLINT   NOT NULL CHECK (Red2 BETWEEN 1 AND 33),
            Red3   SMALLINT   NOT NULL CHECK (Red3 BETWEEN 1 AND 33),
            Red4   SMALLINT   NOT NULL CHECK (Red4 BETWEEN 1 AND 33),
            Red5   SMALLINT   NOT NULL CHECK (Red5 BETWEEN 1 AND 33),
            Red6   SMALLINT   NOT NULL CHECK (Red6 BETWEEN 1 AND 33),
            Blue1  SMALLINT   NOT NULL CHECK (Blue1 BETWEEN 1 AND 16),
            PRIMARY KEY (dNum)
        );
        """)
        cur.execute(
            f"CREATE INDEX IF NOT EXISTS ix_draw_history_ddate "
            f"ON {SCHEMA}.draw_history (dDate);"
        )
        # L3 修复(2026-08-26): 旧表 dNum 为 CHAR(7), 跨 8 位期号年代会截断;
        # 幂等 ALTER 让现有表也升级为 VARCHAR(8)(CREATE TABLE 用 IF NOT EXISTS 不会改已有表)
        cur.execute(
            f"ALTER TABLE {SCHEMA}.draw_history ALTER COLUMN dNum TYPE VARCHAR(8);"
        )
        conn.commit()


# --------------------------------------------------------------------------- #
# T2: 1.csv 导入（幂等：先 DELETE 旧数据再批量 INSERT）
# --------------------------------------------------------------------------- #
def import_draw_history(csv_path, conn: psycopg.Connection) -> int:
    """读取 1.csv（CRLF+BOM 兼容），DELETE 旧 draw_history，executemany 重导入。

    返回导入行数（应 = 3488）。行数/首尾行由调用方或测试校验。
    """
    csv_path = Path(csv_path)
    # utf-8-sig 自动剥离 BOM；CRLF 由 pandas 原生兼容
    df = pd.read_csv(csv_path, encoding="utf-8-sig")
    df.columns = [str(c).strip() for c in df.columns]

    missing = [c for c in DRAW_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(f"CSV 缺少必要列: {missing}")

    # 类型规整：dNum 保留定长7位（前导零语义）；数值列转 int；dDate 转 date
    df["dNum"] = df["dNum"].apply(lambda x: f"{int(x):07d}")
    for c in ["yNum", "mNum", "Red1", "Red2", "Red3", "Red4", "Red5", "Red6", "Blue1"]:
        df[c] = df[c].astype(int)
    df["dDate"] = pd.to_datetime(df["dDate"]).dt.date

    rows = [tuple(r) for r in df[DRAW_COLUMNS].itertuples(index=False, name=None)]
    n = len(rows)

    cols = ",".join(DRAW_COLUMNS)
    placeholders = ",".join(["%s"] * len(DRAW_COLUMNS))
    with conn.cursor() as cur:
        cur.execute(f"TRUNCATE {SCHEMA}.draw_stats, {SCHEMA}.draw_history RESTART IDENTITY;")  # 幂等重导入，先清奖金再清开奖，避免外键级联失败
        cur.executemany(
            f"INSERT INTO {SCHEMA}.draw_history ({cols}) VALUES ({placeholders});",
            rows,
        )
        conn.commit()
    return n


# --------------------------------------------------------------------------- #
# T3: model_predictions 增加 data_date 列（两阶段：可空 -> 回填 -> 默认 -> 非空）
# --------------------------------------------------------------------------- #
def ensure_model_predictions_data_date(conn: psycopg.Connection) -> None:
    """为 ssq.model_predictions 增加 data_date DATE 列并回填存量行（北京日期）。幂等。"""
    with conn.cursor() as cur:
        cur.execute("""
        DO $$
        BEGIN
            IF NOT EXISTS (
                SELECT 1 FROM information_schema.columns
                WHERE table_schema='ssq'
                  AND table_name='model_predictions'
                  AND column_name='data_date'
            ) THEN
                ALTER TABLE ssq.model_predictions ADD COLUMN data_date DATE;
                -- 存量行用 run_at 的日期回填。run_at 由 Python datetime.now()（服务器=Asia/Shanghai）
                -- 写入，存储为无时区的时间戳（即北京墙钟），故 run_at::date 即为北京日期。
                -- 注意：切勿用 (run_at AT TIME ZONE 'Asia/Shanghai')::date —— 该表达式对 timestamptz
                -- 取 ::date 时会按 PG 会话时区(Etc/UTC)结算，反而得到 UTC 日期(晚8h)。
                UPDATE ssq.model_predictions
                   SET data_date = run_at::date
                 WHERE data_date IS NULL;
                ALTER TABLE ssq.model_predictions ALTER COLUMN data_date SET DEFAULT (now() AT TIME ZONE 'Asia/Shanghai')::date;
                ALTER TABLE ssq.model_predictions ALTER COLUMN data_date SET NOT NULL;
            END IF;
        END $$;
        """)
        # 幂等修正：列已存在时上面的 DO 块不执行，DEFAULT 可能仍是 UTC 的 CURRENT_DATE。
        # 统一为北京日期语义（与 schema 注释约定一致），每次 ensure 都执行。
        cur.execute(
            "ALTER TABLE ssq.model_predictions "
            "ALTER COLUMN data_date SET DEFAULT (now() AT TIME ZONE 'Asia/Shanghai')::date;"
        )
        cur.execute(
            f"CREATE INDEX IF NOT EXISTS ix_model_predictions_data_date "
            f"ON {SCHEMA}.model_predictions (data_date);"
        )
        conn.commit()


def draw_history_drift(conn: psycopg.Connection) -> dict:
    """检测 ssq.draw_history 与 1.csv 是否漂移（RISK-3 护栏）。

    只读检查，不写库。返回 {"csv_rows","pg_rows","last_csv_issue","last_pg_issue",
    "drift":bool}。H1 决策：draw_history 为只读归档，由 import_draw_history 手动
    同步；此函数用于运维巡检，发现 pg 落后 csv 时人工跑 sync_draw_history()。
    """
    csv_path = Path(__file__).resolve().parent / "ml/data/1.csv"
    info = {"csv_rows": 0, "pg_rows": 0, "last_csv_issue": None,
            "last_pg_issue": None, "drift": False}
    if csv_path.exists():
        import csv as _csv
        with csv_path.open("r", encoding="utf-8-sig", newline="") as f:
            rows = [r for r in _csv.reader(f) if r and r[0].strip().lower() not in ("dnum", "deliver number")]
        info["csv_rows"] = len(rows)
        if rows:
            info["last_csv_issue"] = rows[-1][0]
    with conn.cursor() as cur:
        cur.execute("SELECT COUNT(*) FROM ssq.draw_history;")
        info["pg_rows"] = cur.fetchone()[0]
        cur.execute("SELECT dNum FROM ssq.draw_history WHERE dNum ~ '^\\d{7}$' ORDER BY (dNum::bigint) DESC LIMIT 1;")
        row = cur.fetchone()
        if row is not None:
            info["last_pg_issue"] = row[0]
    info["drift"] = (info["csv_rows"] != info["pg_rows"])
    return info


def sync_draw_history(conn: psycopg.Connection) -> int:
    """幂等重导 1.csv -> ssq.draw_history（覆盖式）。供运维在 drift 时手动调用。

    不进生产 cron（H1：只读归档 + 人工同步）。
    """
    n = import_draw_history(Path(__file__).resolve().parent / "ml/data/1.csv", conn)
    return n


if __name__ == "__main__":
    # 直接运行本模块即执行 T1+T2+T3 并对结果做基本 sanity 校验
    c = get_conn()
    try:
        ensure_draw_history_schema(c)
        n = import_draw_history(Path(__file__).resolve().parent / "ml/data/1.csv", c)
        print(f"[import] draw_history rows = {n} (expect {EXPECTED_ROWS})")
        ensure_model_predictions_data_date(c)
        d = draw_history_drift(c)
        print(f"[drift] csv={d['csv_rows']} pg={d['pg_rows']} "
              f"last_csv={d['last_csv_issue']} last_pg={d['last_pg_issue']} "
              f"drift={d['drift']}")
        with c.cursor() as cur:
            cur.execute("SELECT COUNT(*) FROM ssq.model_predictions WHERE data_date IS NULL;")
            print("[check] model_predictions data_date NULL =", cur.fetchone()[0])
    finally:
        c.close()
