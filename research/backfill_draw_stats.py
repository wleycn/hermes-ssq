#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""B 档 B3: 从 CWL 官方接口全量回补 ssq.draw_stats 奖金数据 (2013-01-01 起)。
只读接口 + 仅写 draw_stats 新表; 对 draw_history 无对应 dnum 的期号跳过。
幂等: ON CONFLICT DO UPDATE。

用法:
  python3 research/backfill_draw_stats.py --limit 5     # 试跑前5期
  python3 research/backfill_draw_stats.py               # 全量回补
"""
from __future__ import annotations

import argparse
import json
import sys
import urllib.request
from pathlib import Path

HERE = Path(__file__).resolve().parent  # research/
ROOT = HERE.parent  # python/SSQ/
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "ml" / "data"))
sys.path.insert(0, str(ROOT / "ml"))

UA = ("Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
      "(KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36")

CWL_URL = ("https://www.cwl.gov.cn/cwl_admin/front/cwlkj/search/kjxx/findDrawNotice"
           "?name=ssq&pageNo=1&pageSize=3000")


def fetch_all() -> list[dict]:
    req = urllib.request.Request(
        CWL_URL, headers={"User-Agent": UA, "Referer": "https://www.cwl.gov.cn/ygkj/kjgg/"})
    with urllib.request.urlopen(req, timeout=30) as r:
        data = json.loads(r.read().decode("utf-8"))
    return data.get("result", [])


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=0, help="试跑: 只处理前 N 期")
    args = ap.parse_args()

    import db_draw as db
    conn = db.connect()
    try:
        db.ensure_stats_table(conn)
        # 已有 dnum 集合(来自 draw_history, FK 约束)
        with conn.cursor() as cur:
            cur.execute("SELECT dnum FROM ssq.draw_history")
            valid = {r[0] for r in cur.fetchall()}

        results = fetch_all()
        if args.limit:
            results = results[:args.limit]
        print(f"接口返回 {len(results)} 期, 回补其中 draw_history 存在的期...")

        done, skipped_missing, skipped_empty = 0, 0, 0
        for res in results:
            dnum = str(res.get("code", "")).strip()
            if dnum not in valid:
                skipped_missing += 1
                continue
            sales = int(res.get("sales", 0) or 0)
            pool = int(res.get("poolmoney", 0) or 0)
            pg = res.get("prizegrades") or []
            if not (sales or pool or pg):
                skipped_empty += 1
                continue
            db.upsert_draw_stats(conn, {
                "dNum": dnum, "sales": sales,
                "poolmoney": pool, "prizegrades": pg,
            })
            done += 1
        print(f"[ok] 回补完成: 写入={done}, 跳过(无draw_history)={skipped_missing}, "
              f"跳过(空奖金)={skipped_empty}")
    finally:
        conn.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
