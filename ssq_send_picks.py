#!/usr/bin/env python3
# CRON 绑定: 壳=~/.hermes/scripts/ssq_send_picks.py  cron job=ced57f0994d8 (SSQ 发下期预测, 经 ssq-lottery-pipeline skill 触发)
# 本文件是逻辑真身; 改这里即生效, 勿改壳里的副本
"""SSQ 下一期推荐号码：生成 → 落库 PG 复盘 → 邮件发送。

流程:
1. 读 PG ssq.model_predictions 最新 run_at 集成概率
2. 生成旋转矩阵 Wheel 多尺寸(纯 wheel, 无 Top5 锚): 默认 W10(10注) + W20(20注), 各自独立生成
3. 落库 ssq.predicted_picks(复盘表, 幂等: 同 period+mode+wheel_notes 先删后插)
4. 单封邮件: A 区=Wheel10, B 区=Wheel20, 零"锚/结合"过时描述
5. smtplib 直发邮件到 wleycn@163.com(绕过 hermes send email bug)

注: 模型概率来自最近一次 run(未重训), 不同期仅 seed 不同导致采样差异。
"""

import argparse
import json
import os
import subprocess
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import psycopg

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
# 项目根(本项目模块 select_numbers / ml 所在)
PROJECT_ROOT = "/home/hermes/workspace/python/SSQ"
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)
import select_numbers as sn
from ml.popularity import combo_popularity
# 镜像导出: 落库后自动把当期权重进 data-center/ssq/picks (PG 单真源, csv 为镜像备份)
from export_picks_to_datacenter import export_period

DEFAULT_PERIOD = "2026094"
DEFAULT_SEED = 2026094

DDL = """
CREATE TABLE IF NOT EXISTS ssq.predicted_picks (
    id          BIGSERIAL PRIMARY KEY,
    period      TEXT        NOT NULL,           -- 期号, 如 2026094
    run_at      TIMESTAMP   NOT NULL,           -- 生成时间
    mode        TEXT        NOT NULL,           -- 'wheel'
    group_idx   INT         NOT NULL,           -- 组序号(1-based)
    reds        TEXT        NOT NULL,           -- 红球 '9,11,13,15,20,24'
    blue        INT         NOT NULL,           -- 蓝球
    popularity  DOUBLE PRECISION,               -- 组合流行度 [0,1]
    seed        INT         NOT NULL,
    pool_size   INT,                            -- wheel 模式红球池大小
    pass_rate   DOUBLE PRECISION,               -- wheel 模式 6-子集通过率
    n_notes     INT,                            -- wheel 模式总注数
    wheel_notes INT                             -- 尺寸预算: 10/20/30 (区分 W10/W20/W30)
);
CREATE INDEX IF NOT EXISTS idx_predicted_picks_period ON ssq.predicted_picks(period);
"""
# 列迁移兜底: 旧表无 wheel_notes 列时补齐 (幂等)
ALTER_DDL = "ALTER TABLE ssq.predicted_picks ADD COLUMN IF NOT EXISTS wheel_notes INT;"


def db_clear_mode(conn, period, mode, wheel_notes=None):
    """落库前整批清空该 period+mode(+wheel_notes)旧行. wheel_notes=None 时清该 mode 全部尺寸."""
    with conn.cursor() as cur:
        if wheel_notes is None:
            cur.execute(
                "DELETE FROM ssq.predicted_picks WHERE period=%s AND mode=%s",
                (period, mode))
        else:
            cur.execute(
                "DELETE FROM ssq.predicted_picks WHERE period=%s AND mode=%s AND wheel_notes=%s",
                (period, mode, wheel_notes))
    conn.commit()


def db_upsert(conn, period, run_at, mode, group_idx, reds, blue,
              popularity, seed, pool_size=None, pass_rate=None, n_notes=None,
              wheel_notes=None):
    """插入一行预测. 同 period+mode+group_idx+wheel_notes 先删(幂等重跑)."""
    with conn.cursor() as cur:
        cur.execute(
            "DELETE FROM ssq.predicted_picks "
            "WHERE period=%s AND mode=%s AND group_idx=%s AND wheel_notes IS NOT DISTINCT FROM %s",
            (period, mode, group_idx, wheel_notes))
        cur.execute(
            """INSERT INTO ssq.predicted_picks
               (period, run_at, mode, group_idx, reds, blue, popularity, seed,
                pool_size, pass_rate, n_notes, wheel_notes)
               VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s)""",
            (period, run_at, mode, group_idx, reds, blue, popularity, seed,
             pool_size, pass_rate, n_notes, wheel_notes))
    conn.commit()


SEND_EMAIL_CLI = Path.home() / "workspace/ng/skills/common/send-email/send_email.py"


def send_email(subject, html_body):
    """统一收件: 调 send_email.py 中枢 (To=126 + Cc=163 由 .env 兜底, 见中枢)."""
    tmp = Path("/tmp/ssq_picks_body.html")
    tmp.write_text(html_body, encoding="utf-8")
    cmd = [sys.executable, str(SEND_EMAIL_CLI),
           "--subject", subject, "--body-file", str(tmp), "--html"]
    try:
        cp = subprocess.run(cmd, capture_output=True, text=True, timeout=60)
        if cp.returncode != 0:
            print(f"[email] 失败: {cp.stderr.strip()[:200]}")
        else:
            print(f"[email] 已提交中枢: {subject!r}")
    except Exception as e:
        print(f"[email] 异常: {e}")


def render_html(period, seed, run_at, wheels: dict):
    """组装单封邮件: A 区=Wheel10, B 区=Wheel20, 纯 wheel 无锚描述。
    wheels: {10: (tickets, cov), 20: (tickets, cov)} 按尺寸键排序输出。"""
    def rows_html(tickets):
        return "".join(
            f"<tr><td>{i+1}</td><td>{' '.join(f'{r:02d}' for r in t['reds'])}</td>"
            f"<td>{t['blue']:02d}</td><td>{t['popularity']:.3f}</td></tr>"
            for i, t in enumerate(sorted(tickets, key=lambda x: x['popularity'], reverse=True)))

    section_labels = {10: "A", 20: "B", 30: "C"}
    sections = ""
    for N in sorted(wheels.keys()):
        tickets, cov = wheels[N]
        label = section_labels.get(N, str(N))
        sections += f"""
<h3>{label}. 旋转矩阵 Wheel {N} 注（红球池 18 / 共 {N} 注）</h3>
<table border="1" cellpadding="4" cellspacing="0">
<tr><th>#</th><th>红球</th><th>蓝球</th><th>流行度</th></tr>{rows_html(tickets)}
</table>
<p>覆盖率: 4-子集 {cov['four_subset_coverage']*100:.2f}% ｜ 6-子集通过率 {cov['pass_rate']*100:.2f}% ｜ 注数 {cov['n_notes']} ｜ 池大小 {cov.get('pool_size','-')} ｜ 收敛 {cov['converged']}</p>
<p>说明: 若 6 个奖号全部落在池内(概率 C(18,6)/C(33,6)≈17.7%), {N} 注中至少一注中 4 红以上的概率为 {cov['pass_rate']*100:.2f}%。</p>
"""
    return f"""<html><body style="font-family:sans-serif">
<h2>双色球 {period} 期推荐（下一期开奖）</h2>
<p>数据源: 模型集成概率 run_at={run_at}；生成 seed={seed}（可复现）</p>
{sections}
<p><b>诚实声明:</b> 双色球为独立随机抽取, 任何方法无法突破随机下限(每注期望命中 ≈1.09 红/注)。本推荐仅以旋转矩阵提升"若奖号落池则中 4 红以上"的覆盖概率, 并不提升中奖本身概率。以上方案仅供娱乐参考。</p>
</body></html>"""


def compute_next_period(csv_path: str = None) -> tuple[str, int, str | None]:
    """自动算下一期期号: 读 1.csv 最新一行 dNum, 下一期 = dNum+1。
    返回 (period_str, seed_int, latest_date_str)。"""
    import csv
    if csv_path is None:
        csv_path = str(Path(__file__).resolve().parent / "ml" / "data" / "1.csv")
    proj = Path("/home/hermes/workspace/python/SSQ/ml/data/1.csv")
    if proj.exists():
        csv_path = str(proj)
    latest_dnum = None
    latest_date = None
    with open(csv_path, encoding="utf-8-sig", newline="") as f:
        rows = list(csv.DictReader(f))
    if rows:
        last = rows[-1]
        latest_dnum = int(last["dNum"])
        latest_date = last.get("dDate") or last.get("draw_date")
    if latest_dnum is None:
        raise RuntimeError(f"无法从 {csv_path} 推断最新期号")
    nxt = latest_dnum + 1
    return str(nxt), nxt, latest_date


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--period", default=None, help="期号(手动覆盖); 默认自动算下一期")
    ap.add_argument("--seed", type=int, default=None, help="生成 seed; 默认=期号")
    ap.add_argument("--auto-next", action="store_true", default=True,
                    help="自动从 1.csv 算下一期期号(默认开)")
    ap.add_argument("--no-auto-next", dest="auto_next", action="store_false",
                    help="关闭自动算期, 必须显式 --period")
    ap.add_argument("--wheels", type=str, default="10,20",
                    help="多尺寸逗号分隔, 如 '10,20'; 单封邮件 A/B 区各对应一个尺寸")
    ap.add_argument("--no-email", action="store_true",
                    help="不发送邮件(仅落库+打印, 用于测试/验证)")
    ap.add_argument("--dry-run", action="store_true",
                    help="不落库不发送, 仅打印合并统计(纯验证)")
    ap.add_argument("--no-shrink", action="store_true",
                    help="关闭 James-Stein 收缩(默认开启, 概率诚实化; 仅调试用)")
    ap.add_argument("--no-popularity", action="store_true",
                    help="关闭流行度计算(默认开启; 仅调试用)")
    args = ap.parse_args()

    if args.period:
        period, seed = args.period, (args.seed or int(args.period))
    else:
        period, seed_auto, latest_date = compute_next_period()
        seed = args.seed or seed_auto
        print(f"[auto-next] 最新已开奖期={int(period)-1}, 推算下一期={period}"
              f"(本期开奖日期={latest_date})")

    from ml.pg_conn import connect
    conn = connect()  # 凭证从 ~/.hermes/.env 的 DATABASE_URL 读, 不硬编码
    with conn.cursor() as cur:
        cur.execute(DDL)  # 幂等建表
        conn.commit()
        cur.execute(ALTER_DDL)  # 列迁移兜底: 旧表补 wheel_notes 列
        conn.commit()
    # 解析目标尺寸
    sizes = [int(x) for x in args.wheels.split(",") if x.strip()]
    if not sizes:
        sizes = [10, 20]
    print(f"[config] 纯 wheel 尺寸={sizes} (无锚/无结合)")
    # 单一权威入口: 集成概率 → (James-Stein 收缩) → 纯 wheel 多尺寸
    # 研究结论(收缩/无top5锚)只落地在 select_numbers.generate_picks, 此处不重复实现
    run_at, wheels = sn.generate_picks(
        conn, seed, wheels=sizes,
        no_shrink=getattr(args, "no_shrink", False),
        no_popularity=getattr(args, "no_popularity", False),
    )

    now = datetime.now()
    # 清理上一版残留的 top5 锚行(纯 wheel 模式已弃用锚, 旧数据留在库里会污染复盘)
    db_clear_mode(conn, period, "top5")
    for N in sizes:
        tickets, cov = wheels[N]
        print(f"[wheel] N={N} 生成注={len(tickets)} pass_rate={cov['pass_rate']:.4f}")

        if args.dry_run:
            print(f"[dry-run] N={N} 不落库不发送。")
            continue

        # 落库前清空该期+该尺寸旧行(防历史残留 + 尺寸间互不影响)
        db_clear_mode(conn, period, "wheel", wheel_notes=N)
        for i, t in enumerate(tickets, 1):
            db_upsert(conn, period, now, "wheel", i,
                      ",".join(f"{r:02d}" for r in t["reds"]),
                      t["blue"], t["popularity"], seed,
                      pool_size=cov.get("pool_size", 18), pass_rate=cov["pass_rate"],
                      n_notes=cov["n_notes"], wheel_notes=N)

    # 整期落库完成后, 镜像一次到 data-center/ssq/picks (csv 失败仅 log, 不阻断发邮件)
    try:
        export_period(conn, period)
        print(f"[mirror] 已镜像当期权重到 data-center/ssq/picks ({period})")
    except Exception as e:
        print(f"[mirror] 警告: csv 镜像失败(不影响 PG 真源与发信): {e}")

    if args.dry_run:
        conn.close()
        return

    # 单封邮件: A=Wheel10, B=Wheel20, ...
    def _pr(n):
        return f"{wheels.get(n, (None, {}))[1].get('pass_rate', 0)*100:.1f}%"
    sizes_sorted = sorted(wheels.keys())
    title = (f"双色球{period}期推荐: "
             + " + ".join(f"Wheel{n}({_pr(n)})" for n in sizes_sorted)
             + " 双尺寸")
    print(f"[email] 组装单封邮件, 尺寸={sizes_sorted}")
    if not args.no_email:
        send_email(title, render_html(period, seed, run_at, wheels))
    else:
        print(f"[no-email] 跳过发信。标题={title}")

    # 落库确认(全尺寸)
    with conn.cursor() as cur:
        cur.execute("SELECT mode, wheel_notes, COUNT(*) FROM ssq.predicted_picks "
                    "WHERE period=%s GROUP BY mode, wheel_notes ORDER BY 1, 2", (period,))
        print("[db] verify:", cur.fetchall())
    conn.close()


if __name__ == "__main__":
    main()
