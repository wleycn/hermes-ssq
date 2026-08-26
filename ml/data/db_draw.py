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

from ml.pg_conn import pg_dict

PG = pg_dict()  # 凭证从 ~/.hermes/.env 的 DATABASE_URL 读, 不硬编码


def connect() -> Any:
    """建立 PG 连接（类型明确的工厂，避免 **dict 混合类型）。"""
    import psycopg
    return psycopg.connect(host=PG["host"], port=PG["port"], user=PG["user"],
                           password=PG["password"], dbname=PG["dbname"])


def upsert_draw(conn: Any, rec: dict) -> None:
    """幂等写一行开奖到 ssq.draw_history（同 dnum 先删后插）。

    rec 字段: dNum(int), yNum, mNum, dDate('YYYY-MM-DD'),
              Red1..Red6(int), Blue1(int)。
    """
    dnum = str(rec["dNum"]).strip()
    with conn.cursor() as cur:
        cur.execute("DELETE FROM ssq.draw_history WHERE dnum=%s", (dnum,))
        cur.execute(
            """INSERT INTO ssq.draw_history
               (dnum, ynum, mnum, ddate, red1, red2, red3, red4, red5, red6, blue1)
               VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s)""",
            (dnum,
             int(rec["yNum"]), int(rec["mNum"]),
             rec["dDate"],
             int(rec["Red1"]), int(rec["Red2"]), int(rec["Red3"]),
             int(rec["Red4"]), int(rec["Red5"]), int(rec["Red6"]),
             int(rec["Blue1"])),
        )
    conn.commit()


def get_latest_draw(conn: Any) -> Optional[dict]:
    """读最新一期（ORDER BY dnum DESC LIMIT 1）。无数据返回 None。"""
    with conn.cursor() as cur:
        cur.execute(
            """SELECT dnum, ynum, mnum, ddate,
                      red1, red2, red3, red4, red5, red6, blue1
               FROM ssq.draw_history ORDER BY dnum DESC LIMIT 1""")
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
