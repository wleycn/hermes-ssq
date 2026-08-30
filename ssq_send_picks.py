#!/usr/bin/env python3
# CRON 绑定: 壳=~/.hermes/scripts/ssq_send_picks.py  cron job=ced57f0994d8 (SSQ 发下期预测, 经 ssq-lottery-pipeline skill 触发)
# 本文件是逻辑真身; 改这里即生效, 勿改壳里的副本
"""SSQ 下一期推荐号码：生成 → 落库 PG 复盘 → 邮件发送。

流程:
1. 读 PG ssq.model_predictions 最新 run_at 集成概率
2. 生成 A) 5 组(带 popularity 冷门加权, seed=PERIOD)  B) wheel 30 注(池18/30注)
3. 落库 ssq.predicted_picks(复盘表, 幂等: 同 period+mode+group 先删后插)
4. smtplib 直发邮件到 wleycn@163.com(绕过 hermes send email bug)

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
from ml.popularity import combo_popularity, sample_with_popularity

DEFAULT_PERIOD = "2026094"
DEFAULT_SEED = 2026094

DDL = """
CREATE TABLE IF NOT EXISTS ssq.predicted_picks (
    id          BIGSERIAL PRIMARY KEY,
    period      TEXT        NOT NULL,           -- 期号, 如 2026094
    run_at      TIMESTAMP   NOT NULL,           -- 生成时间
    mode        TEXT        NOT NULL,           -- 'top5' | 'wheel'
    group_idx   INT         NOT NULL,           -- 组序号(1-based)
    reds        TEXT        NOT NULL,           -- 红球 '9,11,13,15,20,24'
    blue        INT         NOT NULL,           -- 蓝球
    popularity  DOUBLE PRECISION,               -- 组合流行度 [0,1]
    seed        INT         NOT NULL,
    pool_size   INT,                            -- wheel 模式红球池大小
    pass_rate   DOUBLE PRECISION,               -- wheel 模式 6-子集通过率
    n_notes     INT                             -- wheel 模式注数
);
CREATE INDEX IF NOT EXISTS idx_predicted_picks_period ON ssq.predicted_picks(period);
"""


def load_probs(conn):
    """读最新 run_at 集成概率, 返回 (red_mean, blue_mean, run_at, models)."""
    return sn.load_latest_probs(conn)


def gen_top5(red_mean, blue_mean, rng):
    """5 组: 红球 popularity 冷门加权采样, 蓝球受控随机. 返回带流行度注单列表(流行度降序)."""
    out = []
    for g in range(5):
        reds = sample_with_popularity(red_mean, rng, temperature=0.6,
                                      lambda_=0.3, n_candidates=200)
        blue = int(sn._sample_blue(blue_mean, rng))  # 已 1-indexed
        out.append({"group": g + 1, "reds": [int(n) for n in reds], "blue": blue,
                    "popularity": float(combo_popularity(reds))})
    return sorted(out, key=lambda x: x["popularity"], reverse=True)


def gen_wheel(red_mean, blue_mean, seed, total_notes=20, extra=5):
    """生成 Wheel 候选: Top-18 概率池 → greedy_cover。
    total_notes = 对外结合后总注数(10/20/30); 内部多生成 extra 注作去重余量。
    最终由 merge_top5_wheel 把 Top5 的 5 注作锚嵌入, 结合总注 = total_notes(非 5+total 叠加)。"""
    res = sn.build_wheel_tickets(red_mean, blue_mean, pool_size=18,
                                 max_notes=total_notes + extra, restarts=3, seed=seed,
                                 popularity_fn=None, lambda_=0.3)
    return res


def merge_top5_wheel(top5, wheel_tickets, total):
    """结合(非叠加): Top5 的 5 注作为锚必含, 从 Wheel 候选按流行度挑 total-5 注去重补足,
    使最终总注 = total(10/20/30)。预算固定为 N, 而非 5+N 的简单叠加——这才是'结合'的意义:
    用 Top5 的冷门加权智慧播种 Wheel 的覆盖, 而非两份独立票相加。

    返回 merged 列表(长度=total; 前 len(top5) 为 Top5 锚, 其后为 Wheel 补足)。
    若 Wheel 去重后可用注不足 total-5(罕见, extra 余量足够时不会发生), 则取全部可用注。"""
    if total < len(top5):
        raise ValueError(f"total_notes({total}) 必须 ≥ Top5 注数({len(top5)})")
    out = list(top5)
    seen = {(tuple(sorted(g["reds"])), g["blue"]) for g in top5}
    for t in sorted(wheel_tickets, key=lambda x: x["popularity"], reverse=True):
        if len(out) >= total:
            break
        k = (tuple(sorted(t["reds"])), t["blue"])
        if k in seen:
            continue
        seen.add(k)
        out.append(t)
    return out


def db_clear_mode(conn, period, mode):
    """落库前整批清空该 period+mode 旧行(防历史 wheel30 残留 group_idx>新注数)."""
    with conn.cursor() as cur:
        cur.execute(
            "DELETE FROM ssq.predicted_picks WHERE period=%s AND mode=%s",
            (period, mode))
    conn.commit()


def db_upsert(conn, period, run_at, mode, group_idx, reds, blue,
              popularity, seed, pool_size=None, pass_rate=None, n_notes=None):
    """插入一行预测. 同 period+mode+group_idx 先删(幂等重跑)."""
    with conn.cursor() as cur:
        cur.execute(
            "DELETE FROM ssq.predicted_picks WHERE period=%s AND mode=%s AND group_idx=%s",
            (period, mode, group_idx))
        cur.execute(
            """INSERT INTO ssq.predicted_picks
               (period, run_at, mode, group_idx, reds, blue, popularity, seed,
                pool_size, pass_rate, n_notes)
               VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s)""",
            (period, run_at, mode, group_idx, reds, blue, popularity, seed,
             pool_size, pass_rate, n_notes))
    conn.commit()


SEND_EMAIL_CLI = Path.home() / "workspace/ng/skills/common/send-email/send_email.py"


def send_email(subject, html_body):
    """统一收件: 调 send_email.py 中枢 (To=126 + Cc=163 由 .env 兜底, 见中枢).

    说明: 早期版本内嵌 smtplib 硬编码 163 并称'绕过 hermes send email bug';
    现 hermes 发信通道已验证可用, 收敛到统一中枢, 不再各自维护 smtp 逻辑.
    """
    # 临时落盘 html 正文, 交给中枢发送 (HTML 模式)
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


def render_html(period, seed, run_at, top5, wheel, dropped=0, wheel_notes=20, merged_total=None):
    """组装邮件 HTML: Top5 锚(优先) + Wheel 补足(去重后并集) + 覆盖率报告."""
    res = wheel  # build_wheel_tickets 返回 dict
    rows5 = "".join(
        f"<tr><td>{g['group']}</td><td><b>{' '.join(f'{r:02d}' for r in g['reds'])}</b></td>"
        f"<td><b>{g['blue']:02d}</b></td><td>{g['popularity']:.3f}</td></tr>"
        for g in top5)
    cov = res["coverage"]
    wt = res["tickets"]  # 已是去重后的 wheel 补足部分
    rows_w = "".join(
        f"<tr><td>{i+1}</td><td>{' '.join(f'{r:02d}' for r in t['reds'])}</td>"
        f"<td>{t['blue']:02d}</td><td>{t['popularity']:.3f}</td></tr>"
        for i, t in enumerate(sorted(wt, key=lambda x: x["popularity"], reverse=True)))
    total = merged_total if merged_total is not None else (len(top5) + len(wt))
    return f"""<html><body style="font-family:sans-serif">
<h2>双色球 {period} 期推荐（下一期开奖）</h2>
<p>数据源: 模型集成概率 run_at={run_at}；生成 seed={seed}（可复现）</p>

<h3>A. Top5 锚（红球冷门加权 + 蓝球受控随机，作为结合必含项）</h3>
<table border="1" cellpadding="4" cellspacing="0">
<tr><th>组</th><th>红球</th><th>蓝球</th><th>流行度</th></tr>{rows5}
</table>
<p>注: 流行度越低越冷门(避开连号/生日号/全奇偶等热门组合)；仅供娱乐参考。</p>

<h3>B. 旋转矩阵 Wheel 补足（池18 / 候选 {wheel_notes}注，与 A 去重并集至总预算）</h3>
<table border="1" cellpadding="4" cellspacing="0">
<tr><th>#</th><th>红球</th><th>蓝球</th><th>流行度</th></tr>{rows_w}
</table>
<p>覆盖率: 4-子集 {cov['four_subset_coverage']*100:.2f}% ｜ 6-子集通过率 {cov['pass_rate']*100:.2f}% ｜ 注数 {cov['n_notes']} ｜ 池大小 {cov.get('pool_size','-')} ｜ 收敛 {cov['converged']}</p>
<p>说明: 若 6 个奖号全部落在池内(概率 C(18,6)/C(33,6)≈17.7%)，{wheel_notes} 注中至少一注中 4 红以上的概率为 {cov['pass_rate']*100:.2f}%。</p>
<p><b>结合总注数 = {total}（A 锚 5 + B 补足 {len(wt)}；Top5 已嵌入而非叠加，预算固定为 {wheel_notes}）。</b></p>
<p>诚实声明: 四法同吃 FLAT 概率, 每注命中率均=随机下限(≈1.09红/注); 覆盖差异只在规模/成本, 不在精度。仅供娱乐参考。</p>
</body></html>"""


def compute_next_period(csv_path: str = None) -> tuple[str, int, str | None]:
    """自动算下一期期号: 读 1.csv 最新一行 dNum, 下一期 = dNum+1。

    返回 (period_str, seed_int, latest_date_str)。
    latest_date_str 为本期开奖日期, 供状态门判断'本期是否已开奖'。
    """
    import csv
    if csv_path is None:
        csv_path = str(Path(__file__).resolve().parent / "ml" / "data" / "1.csv")
    # 注意: 脚本在 ~/.hermes/scripts/, 项目在 /home/hermes/workspace/python/SSQ
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
    ap.add_argument("--wheel-notes", type=int, default=20,
                    help="wheel 注数(默认20; 10/20/30 任配, walk-forward 验 20 为甜点)")
    ap.add_argument("--no-email", action="store_true",
                    help="不发送邮件(仅落库+打印, 用于测试/验证)")
    ap.add_argument("--dry-run", action="store_true",
                    help="不落库不发送, 仅打印合并统计(纯验证)")
    args = ap.parse_args()

    if args.period:
        period, seed = args.period, (args.seed or int(args.period))
    else:
        period, seed_auto, latest_date = compute_next_period()
        seed = args.seed or seed_auto
        # 状态门: 打印本期开奖日期, 供人工核对(本期应已由抓开奖 cron 入库)
        print(f"[auto-next] 最新已开奖期={int(period)-1}, 推算下一期={period}"
              f"(本期开奖日期={latest_date})")

    from ml.pg_conn import connect
    conn = connect()  # 凭证从 ~/.hermes/.env 的 DATABASE_URL 读, 不硬编码
    with conn.cursor() as cur:
        cur.execute(DDL)  # 幂等建表
        conn.commit()
    red_mean, blue_mean, run_at, models = load_probs(conn)
    rng = np.random.default_rng(seed)
    top5 = gen_top5(red_mean, blue_mean, rng)
    # 结合预算 = 总注数(10/20/30); 内部多生成 extra 注作锚嵌入余量
    wheel = gen_wheel(red_mean, blue_mean, seed, total_notes=args.wheel_notes, extra=5)

    # ①+②: Top5 的 5 注作锚必含, 从 Wheel 候选补足至总注=args.wheel_notes(结合非叠加)
    merged = merge_top5_wheel(top5, wheel["tickets"], total=args.wheel_notes)
    top5_anchor = merged[:len(top5)]
    wheel_fill = merged[len(top5):]  # Wheel 补足部分(已去重, 不含 Top5 锚)

    now = datetime.now()
    print(f"[combine] period={period} 结合总注={len(merged)} "
          f"(Top5锚{len(top5_anchor)} + Wheel补{len(wheel_fill)}) "
          f"wheel候选池={len(wheel['tickets'])} pass_rate={wheel['coverage']['pass_rate']:.4f} run_at={run_at}")

    if args.dry_run:
        print(f"[dry-run] 不落库不发送。结合总注={len(merged)}。")
        conn.close()
        return

    # 落库前整批清空该期旧行(防历史残留)
    db_clear_mode(conn, period, "top5")
    db_clear_mode(conn, period, "wheel")
    # 落库 A: Top5 锚(5 注, 仍标记 top5 模式便于复盘)
    for g in top5_anchor:
        db_upsert(conn, period, now, "top5", g["group"], ",".join(f"{r:02d}" for r in g["reds"]),
                  g["blue"], g["popularity"], seed)
    # 落库 B: Wheel 补足部分(标记 wheel 模式)
    cov = wheel["coverage"]
    for i, t in enumerate(wheel_fill, 1):
        db_upsert(conn, period, now, "wheel", i, ",".join(f"{r:02d}" for r in t["reds"]),
                  t["blue"], t["popularity"], seed,
                  pool_size=cov.get("pool_size", 18), pass_rate=cov["pass_rate"],
                  n_notes=cov["n_notes"])

    # 打印 + 邮件
    title = (f"双色球{period}期推荐: Top5+Wheel 结合{len(merged)}注"
             f"(通过率{cov['pass_rate']*100:.1f}%)")
    print(f"[db] period={period} 结合总注={len(merged)} "
          f"(top5锚{len(top5_anchor)}+wheel补{len(wheel_fill)}) "
          f"pass_rate={cov['pass_rate']:.4f} run_at={run_at}")
    if not args.no_email:
        send_email(title, render_html(period, seed, run_at, top5_anchor,
                                      {"coverage": cov, "tickets": wheel_fill},
                                      dropped=0, wheel_notes=args.wheel_notes,
                                      merged_total=len(merged)))
    else:
        print(f"[no-email] 跳过发信。标题={title}")
    # 落库确认
    with conn.cursor() as cur:
        cur.execute("SELECT mode, COUNT(*) FROM ssq.predicted_picks WHERE period=%s GROUP BY mode ORDER BY 1", (period,))
        print("[db] verify:", cur.fetchall())
    conn.close()


if __name__ == "__main__":
    main()
