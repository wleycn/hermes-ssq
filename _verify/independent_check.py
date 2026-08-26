#!/usr/bin/env python3
"""Independent cross-verification (TestAutomationEngineer) — direct SQL + pandas,
no reliance on Dev's functions. Evidence printed as JSON to stdout."""
from __future__ import annotations
import json, sys
from datetime import date, datetime
from pathlib import Path

import psycopg
import pandas as pd

SSQ = Path("/home/hermes/workspace/python/SSQ")
CSV = SSQ / "ml/data/1.csv"
from ml.pg_conn import pg_dict
PG = pg_dict()  # 凭证从 ~/.hermes/.env 的 DATABASE_URL 读

ev = {}

c = psycopg.connect(**PG)
try:
    cur = c.cursor()

    # 1. draw_history count
    cur.execute("SELECT COUNT(*) FROM ssq.draw_history;")
    dh_count = cur.fetchone()[0]
    ev["draw_history_count"] = dh_count

    # 2. first / last rows
    cur.execute("SELECT dNum, dDate FROM ssq.draw_history ORDER BY dDate ASC, dNum ASC LIMIT 1;")
    first = cur.fetchone()
    cur.execute("SELECT dNum, dDate FROM ssq.draw_history ORDER BY dDate DESC, dNum DESC LIMIT 1;")
    last = cur.fetchone()
    ev["first_row"] = {"dNum": first[0], "dDate": str(first[1])}
    ev["last_row"] = {"dNum": last[0], "dDate": str(last[1])}

    # 3. distinct dNum
    cur.execute("SELECT COUNT(DISTINCT dNum) FROM ssq.draw_history;")
    ev["distinct_dNum"] = cur.fetchone()[0]

    # 4. independent read of 1.csv
    df = pd.read_csv(CSV, encoding="utf-8-sig")
    df.columns = [str(x).strip() for x in df.columns]
    ev["csv_row_count"] = len(df)
    # 1.csv dNum may be int without leading zeros; normalize as 7-digit
    df["dNum7"] = df["dNum"].apply(lambda x: f"{int(x):07d}")
    # find 2026001
    mid = df[df["dNum7"] == "2026001"]
    ev["csv_has_2026001"] = len(mid) == 1
    if len(mid) == 1:
        r = mid.iloc[0]
        csv_mid = {
            "dNum": "2026001",
            "yNum": int(r["yNum"]), "mNum": int(r["mNum"]),
            "dDate": str(pd.to_datetime(r["dDate"]).date()),
            "Red1": int(r["Red1"]), "Red2": int(r["Red2"]), "Red3": int(r["Red3"]),
            "Red4": int(r["Red4"]), "Red5": int(r["Red5"]), "Red6": int(r["Red6"]),
            "Blue1": int(r["Blue1"]),
        }
        ev["csv_2026001"] = csv_mid
        cur.execute(
            "SELECT dNum,yNum,mNum,dDate,Red1,Red2,Red3,Red4,Red5,Red6,Blue1 "
            "FROM ssq.draw_history WHERE dNum=%s;", ("2026001",)
        )
        pg_mid = cur.fetchone()
        pg_dict = {
            "dNum": pg_mid[0], "yNum": int(pg_mid[1]), "mNum": int(pg_mid[2]),
            "dDate": str(pg_mid[3]),
            "Red1": int(pg_mid[4]), "Red2": int(pg_mid[5]), "Red3": int(pg_mid[6]),
            "Red4": int(pg_mid[7]), "Red5": int(pg_mid[8]), "Red6": int(pg_mid[9]),
            "Blue1": int(pg_mid[10]),
        }
        ev["pg_2026001"] = pg_dict
        ev["spot_check_2026001_match"] = (csv_mid == pg_dict)

    # 5. data_date column: non-null + distinct values
    cur.execute("SELECT COUNT(*) FROM ssq.model_predictions WHERE data_date IS NULL;")
    ev["mp_data_date_null"] = cur.fetchone()[0]
    cur.execute("SELECT data_date, COUNT(*) FROM ssq.model_predictions GROUP BY 1 ORDER BY 1;")
    ev["mp_data_date_distinct"] = [{"data_date": str(d), "n": n} for d, n in cur.fetchall()]
    # all should equal today's Beijing date 2026-08-13
    today = datetime.now().date()
    ev["server_now_date"] = str(today)
    cur.execute("SELECT COUNT(*) FROM ssq.model_predictions WHERE data_date = %s;", (today,))
    ev["mp_data_date_eq_today"] = cur.fetchone()[0]
    cur.execute("SELECT COUNT(*) FROM ssq.model_predictions;")
    ev["mp_total"] = cur.fetchone()[0]

    # 6. production rows: data_date == run_at::date (Beijing) for all
    cur.execute(
        "SELECT COUNT(*) FROM ssq.model_predictions "
        "WHERE data_date IS DISTINCT FROM run_at::date;"
    )
    ev["mp_mismatch_runat"] = cur.fetchone()[0]

    # 7. confirm column is NOT NULL + default (schema sanity)
    cur.execute(
        "SELECT column_name, is_nullable, column_default FROM information_schema.columns "
        "WHERE table_schema='ssq' AND table_name='model_predictions' AND column_name='data_date';"
    )
    col = cur.fetchone()
    ev["mp_data_date_col_def"] = {"is_nullable": col[1], "default": col[2]} if col else None
finally:
    c.close()

print(json.dumps(ev, ensure_ascii=False, indent=2))
