#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""双色球开奖表 ssq.draw_history 读写（与 update_ssq.py 配套）。

职责（Rocky 2026-08-26 拍板）:
  开奖获取 → 写 CSV → 写 ssq.draw_history → 从本表读最新 → 发邮件(带号码)。
本模块只管开奖表的 upsert 与 latest 读取，不碰采集/邮件。
连接约定与 reconcile_picks.PG 一致。
"""
from __future__ import annotations

from typing import Any, Optional

from ml.pg_conn import connect as pg_connect, pg_dict

PG = pg_dict()  # 凭证从 ~/.hermes/.env 的 DATABASE_URL 读, 不硬编码


def connect() -> Any:
    """建立 PG 连接（复用 ml.pg_conn.connect 工厂，统一凭证来源）。"""
    return pg_connect()


def upsert_draw(conn: Any, rec: dict) -> None:
    """幂等写一行开奖到 ssq.draw_history（单条 ON CONFLICT DO UPDATE，原子 upsert）。

    L2 修复(2026-08-26): 原 DELETE+INSERT 两步提交非原子, 崩溃会丢行;
    改为单条 UPSERT, 依赖 PK dNum 保证幂等与原子性。
    rec 字段: dNum(int), yNum, mNum, dDate('YYYY-MM-DD'),
              Red1..Red6(int), Blue1(int)。
    """
    dnum = str(rec["dNum"]).strip()
    with conn.cursor() as cur:
        cur.execute(
            f"""INSERT INTO ssq.draw_history
               (dnum, ynum, mnum, ddate, red1, red2, red3, red4, red5, red6, blue1)
               VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s)
               ON CONFLICT (dnum) DO UPDATE SET
                   ynum=EXCLUDED.ynum, mnum=EXCLUDED.mnum, ddate=EXCLUDED.ddate,
                   red1=EXCLUDED.red1, red2=EXCLUDED.red2, red3=EXCLUDED.red3,
                   red4=EXCLUDED.red4, red5=EXCLUDED.red5, red6=EXCLUDED.red6,
                   blue1=EXCLUDED.blue1""",
            (dnum,
             int(rec["yNum"]), int(rec["mNum"]),
             rec["dDate"],
             int(rec["Red1"]), int(rec["Red2"]), int(rec["Red3"]),
             int(rec["Red4"]), int(rec["Red5"]), int(rec["Red6"]),
             int(rec["Blue1"])),
        )
    conn.commit()


def ensure_stats_table(conn: Any) -> None:
    """幂等建 ssq.draw_stats 表(奖金数据隔离表)。2026-08-29 B 档新增。"""
    with conn.cursor() as cur:
        cur.execute(
            """CREATE TABLE IF NOT EXISTS ssq.draw_stats (
                   dnum        VARCHAR(8) PRIMARY KEY,
                   sales       BIGINT,
                   poolmoney   BIGINT,
                   prizegrades JSONB,
                   FOREIGN KEY (dnum) REFERENCES ssq.draw_history(dnum)
               )""")
    conn.commit()


def upsert_draw_stats(conn: Any, rec: dict) -> None:
    """幂等写一行奖金数据到 ssq.draw_stats（单条 ON CONFLICT DO UPDATE）。

    字段: dnum(str), sales(int, 元), poolmoney(int, 元),
          prizegrades(list[dict{type,typenum,typemoney}] 或 JSON 串)。
    2026-08-29 B 档新增: 奖金数据与开奖号正交, 独立表不影响主流程。
    """
    import json as _json
    dnum = str(rec["dNum"]).strip()
    pg_type = _json.dumps(rec["prizegrades"], ensure_ascii=False) \
        if not isinstance(rec["prizegrades"], str) else rec["prizegrades"]
    with conn.cursor() as cur:
        cur.execute(
            """INSERT INTO ssq.draw_stats (dnum, sales, poolmoney, prizegrades)
               VALUES (%s,%s,%s,%s)
               ON CONFLICT (dnum) DO UPDATE SET
                   sales=EXCLUDED.sales, poolmoney=EXCLUDED.poolmoney,
                   prizegrades=EXCLUDED.prizegrades""",
            (dnum, int(rec["sales"]), int(rec["poolmoney"]), pg_type))
    conn.commit()


def get_latest_draw(conn: Any) -> Optional[dict]:
    """读最新一期（按期号数值排序）。无数据返回 None。"""
    with conn.cursor() as cur:
        cur.execute(
            r"""SELECT dnum, ynum, mnum, ddate,
                      red1, red2, red3, red4, red5, red6, blue1
               FROM ssq.draw_history
               WHERE dnum ~ '^\d{7}$'
               ORDER BY (dnum::bigint) DESC LIMIT 1""")
        row = cur.fetchone()
    if not row:
        return None
    dnum, ynum, mnum, ddate, r1, r2, r3, r4, r5, r6, b = row
    return {
        "dNum": int(dnum), "yNum": int(ynum), "mNum": int(mnum),
        "dDate": ddate.strftime("%Y-%m-%d") if hasattr(ddate, "strftime") else str(ddate),
        "reds": [int(r1), int(r2), int(r3), int(r4), int(r5), int(r6)],
        "blue": int(b),
    }
