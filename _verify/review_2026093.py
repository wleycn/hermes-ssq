#!/usr/bin/env python3
"""2026093 开奖入库 + 复盘（幂等）。
- 把 2026093 开奖号 UPSERT 进 ssq.draw_history（1.csv 已含，PG 缺这期）
- 用已存的 predicted_picks(2026093) vs 开奖号做命中复盘
- 对比随机基线（红球期望 6*6/33≈1.0909，蓝球 1/16）
不修改 1.csv、不重跑模型。
"""
import psycopg
from datetime import date

PG = "postgresql://hermes:hermes123@127.0.0.1:5432/hermes"

# —— 2026093 开奖（中彩网权威核实）——
ISSUE = "2026093"
Y, M = 2026, 8
DDATE = date(2026, 8, 13)
ACTUAL_RED = [5, 8, 15, 20, 21, 24]
ACTUAL_BLUE = 9
ACTUAL_SET = set(ACTUAL_RED)


def upsert_draw(cur):
    cur.execute(
        """INSERT INTO ssq.draw_history
           (dnum,ynum,mnum,ddate,red1,red2,red3,red4,red5,red6,blue1)
           VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s)
           ON CONFLICT (dnum) DO UPDATE SET
             ynum=EXCLUDED.ynum, mnum=EXCLUDED.mnum, ddate=EXCLUDED.ddate,
             red1=EXCLUDED.red1,red2=EXCLUDED.red2,red3=EXCLUDED.red3,
             red4=EXCLUDED.red4,red5=EXCLUDED.red5,red6=EXCLUDED.red6,
             blue1=EXCLUDED.blue1""",
        (ISSUE, Y, M, DDATE, *ACTUAL_RED, ACTUAL_BLUE),
    )


def grade(red_hit: int, blue_hit: bool) -> str:
    if red_hit == 6 and blue_hit:
        return "一等奖(6+1)"
    if red_hit == 6:
        return "二等奖(6+0)"
    if red_hit == 5 and blue_hit:
        return "三等奖(5+1)"
    if red_hit == 5 or (red_hit == 4 and blue_hit):
        return "四等奖(5+0 / 4+1)"
    if red_hit == 4 or (red_hit == 3 and blue_hit):
        return "五等奖(4+0 / 3+1)"
    if red_hit <= 2 and blue_hit:
        return "六等奖(2+1 / 1+1 / 0+1)"
    return "未中奖"


def main():
    conn = psycopg.connect(PG)
    cur = conn.cursor()
    upsert_draw(cur)
    conn.commit()
    print(f"[OK] draw_history UPSERT {ISSUE} 红{ACTUAL_RED} 蓝{ACTUAL_BLUE}")

    cur.execute(
        "SELECT mode,group_idx,reds,blue FROM ssq.predicted_picks WHERE period=%s ORDER BY mode,group_idx",
        (ISSUE,),
    )
    rows = cur.fetchall()
    conn.close()

    red_hits, blue_hits = [], 0
    top5, wheel = [], []
    grade_count = {}
    for mode, g, reds_s, blue in rows:
        pred_red = [int(x) for x in reds_s.split(",")]
        rh = len(set(pred_red) & ACTUAL_SET)
        bh = blue == ACTUAL_BLUE
        red_hits.append(rh)
        if bh:
            blue_hits += 1
        g_ = grade(rh, bh)
        grade_count[g_] = grade_count.get(g_, 0) + 1
        entry = (g, pred_red, blue, rh, bh, g_)
        (top5 if mode == "top5" else wheel).append(entry)

    n = len(rows)
    avg_red = sum(red_hits) / n
    blue_rate = blue_hits / n

    print(f"\n=== 复盘 {ISSUE}（共 {n} 注预测）===")
    print(f"开奖红球: {ACTUAL_RED}  蓝球: {ACTUAL_BLUE}")
    print(f"红球平均命中: {avg_red:.3f}  (随机基线≈1.091)")
    print(f"蓝球命中注数: {blue_hits}/{n} = {blue_rate*100:.1f}%  (随机基线 6.25%)")
    print("\n等级分布:")
    for k, v in sorted(grade_count.items(), key=lambda x: -x[1]):
        print(f"  {k}: {v} 注")

    print("\n[top5] 5 组命中明细:")
    for g, pr, bl, rh, bh, g_ in top5:
        print(f"  组{g}: 红{pr} 蓝{bl} -> 红中{rh} 蓝{'中' if bh else '否'} [{g_}]")
    print("\n[wheel] 30 注红球命中分布:")
    rh_wheel = [e[3] for e in wheel]
    from collections import Counter
    for k in sorted(Counter(rh_wheel)):
        print(f"  红中{k}: {Counter(rh_wheel)[k]} 注")
    wb = [e for e in wheel if e[4]]
    print(f"  wheel 蓝球命中: {len(wb)} 注 -> {[(e[0],e[2]) for e in wb]}")


if __name__ == "__main__":
    main()
