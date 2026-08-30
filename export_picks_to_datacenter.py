"""
SSQ 选号导出: PG ssq.predicted_picks -> data-center/ssq/picks/[SPICK][date]-<期号>.csv

真源原则: 选号真源在 PG(pg_schema.pg_dict()), data-center/ssq/picks 是它的归档快照。
用法:
  .venv/bin/python export_picks_to_datacenter.py            # 导出全部期号(幂等覆盖)
  .venv/bin/python export_picks_to_datacenter.py --period 2026094   # 仅一期
"""
from __future__ import annotations

import csv
import sys
from datetime import datetime
from pathlib import Path

# 复用项目 PG 凭证约定(从 ~/.hermes/.env DATABASE_URL 读, 不硬编码)
sys.path.insert(0, str(Path(__file__).resolve().parent))
from pg_schema import get_conn, pg_dict  # noqa: E402

DC_PICKS = Path("/home/hermes/workspace/data-center/ssq/picks")
COLS = [
    "period", "run_at", "mode", "group_idx", "reds", "blue",
    "popularity", "seed", "pool_size", "pass_rate", "n_notes", "wheel_notes",
]


def export_period(conn, period: str) -> Path:
    DC_PICKS.mkdir(parents=True, exist_ok=True)
    out = DC_PICKS / f"[SPICK][{period[:4]}-{period[4:]}][{period}].csv"
    cur = conn.cursor()
    cur.execute(
        "SELECT period, run_at, mode, group_idx, reds, blue, "
        "popularity, seed, pool_size, pass_rate, n_notes, wheel_notes "
        "FROM ssq.predicted_picks WHERE period=%s ORDER BY group_idx",
        (period,),
    )
    rows = cur.fetchall()
    if not rows:
        print(f"[export] 期号 {period} 无记录, 跳过")
        return out
    with out.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(COLS)
        for r in rows:
            w.writerow([r[i] for i in range(len(COLS))])
    print(f"[export] {period}: {len(rows)} 注 -> {out}")
    return out


def main() -> None:
    period = None
    if "--period" in sys.argv:
        period = sys.argv[sys.argv.index("--period") + 1]
    conn = get_conn()
    try:
        if period:
            export_period(conn, period)
        else:
            cur = conn.cursor()
            cur.execute("SELECT DISTINCT period FROM ssq.predicted_picks ORDER BY period")
            for (p,) in cur.fetchall():
                export_period(conn, p)
    finally:
        conn.close()


if __name__ == "__main__":
    print(f"[export] 启动 {datetime.now():%Y-%m-%d %H:%M:%S}")
    main()
