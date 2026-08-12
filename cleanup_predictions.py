#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""清理 ssq.model_predictions 中超过 30 个自然日的预测记录（滚动 30 天，ADR-1/ADR-2）。

重要约束（依据实测修正）：
  - 仅作用于 ssq.model_predictions；ssq.draw_history 是历史开奖归档，永不清理，本脚本绝不触碰。
  - 日期口径统一以 Python 侧 datetime.now()（服务器=Asia/Shanghai 北京时间）为准，
    cutoff = 今天北京日期 - 30 天；不依赖 PG 的 CURRENT_DATE（Etc/UTC，比北京晚 8h）。
  - 默认 --dry-run：只 SELECT COUNT 打印将删数，不执行 DELETE（安全默认）。
  - 显式 --confirm 才真正删除，并打印已删行数。

用法：
  .venv/bin/python cleanup_predictions.py            # dry-run 预览
  .venv/bin/python cleanup_predictions.py --confirm  # 执行删除
"""
from __future__ import annotations

import argparse
import sys
from datetime import datetime, date, timedelta
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from pg_schema import get_conn, SCHEMA  # noqa: E402

KEEP_DAYS = 30


def keep_recent_30d(conn, dry_run: bool = True) -> int:
    """保留最近 KEEP_DAYS 个自然日内的预测，清理更旧的。

    cutoff 在 Python 侧计算（北京时间），通过参数传入 SQL，避免依赖 PG CURRENT_DATE。
    返回：将删除 / 已删除的行数。
    注意：本函数只动 ssq.model_predictions，绝不触碰 ssq.draw_history。
    """
    cutoff = datetime.now().date() - timedelta(days=KEEP_DAYS)
    with conn.cursor() as cur:
        cur.execute(
            f"SELECT COUNT(*) FROM {SCHEMA}.model_predictions WHERE data_date < %s;",
            (cutoff,),
        )
        will_delete = cur.fetchone()[0]

    if dry_run:
        print(f"[dry-run] cutoff={cutoff}  will_delete={will_delete}  （未执行删除，draw_history 不受影响）")
        return will_delete

    with conn.cursor() as cur:
        cur.execute(
            f"DELETE FROM {SCHEMA}.model_predictions WHERE data_date < %s;",
            (cutoff,),
        )
        deleted = cur.rowcount
    conn.commit()
    print(f"[confirm]  cutoff={cutoff}  deleted={deleted}  （仅 model_predictions，draw_history 不受影响）")
    return deleted


def main():
    ap = argparse.ArgumentParser(
        description="清理 ssq.model_predictions 超过 30 天的预测记录（不影响 draw_history）。"
    )
    ap.add_argument("--confirm", action="store_true", help="执行删除；默认仅 dry-run 预览")
    args = ap.parse_args()

    conn = get_conn()
    try:
        keep_recent_30d(conn, dry_run=not args.confirm)
    finally:
        conn.close()


if __name__ == "__main__":
    main()
