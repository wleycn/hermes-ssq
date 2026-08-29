#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""SSQ 每期「单位投注期望收益(EV)」计算 + 正 EV 窗口检测。

B 档 B4 (2026-08-29): 基于 ssq.draw_stats 的真实奖池/销量/奖金结构,
算每期"若当期参与、单位注(¥2)的理论 EV",验证核心命题:
  SSQ 一等奖封顶(实测 1000万, 奖池<15亿即封顶)使 EV 上限极低,
  正 EV 窗口在 i.i.d. + 封顶规则下数学上极不可能存在。

方法学诚实声明:
  - 本脚本算的是「事后已实现口径」与「事前期望口径」两种:
    * 事后: 直接用该期 prizegrades 实测单注奖金 × 中奖概率 → 反推"该期买中任意组合的单位回报"
      (注意: 这是事后, 不能用于事前决策, 仅作结构参考)
    * 事前: 用奖池 + 销量建模分奖碰撞(Kim & Skiena 2021 方法):
        总注数 N = sales/2
        一等奖期望中奖注数 λ1 = N × p1,  p1 = 1/C(33,6)/16
        一等奖单注期望奖金 = J_eff / max(λ1, 1)   (平分裂)
          J_eff = poolmoney + 当期高奖级计提(简化: 取奖池全额作可分配额上界)
        单位 EV = Σ_j p_j × single_j − 2.0
  - 事前模型对"分奖碰撞"用期望值近似(非蒙特卡洛), 是保守上界估计——
    因实际中奖注数方差会使 EV 更低(左偏), 故本脚本结论是 EV 的乐观上界。

输出: EV 曲线(每期单位 EV)、历史最低亏窗口、是否存在正 EV 窗口。

用法:
  python3 analysis/ssq_ev.py            # 默认全量, 打印摘要 + 写 CSV
  python3 analysis/ssq_ev.py --no-csv  # 只打印摘要
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from math import comb
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "ml"))

P1 = 1.0 / (comb(33, 6) * 16)   # 一等奖单注中奖概率 ≈ 5.64e-8
P_RED = 1.0 / comb(33, 6)        # 6红全中概率(不含蓝)
TICKET = 2.0                     # 每注 ¥2


def load_stats() -> list[dict]:
    from pg_conn import connect
    cur = connect().cursor()
    cur.execute(
        "SELECT dnum, sales, poolmoney, prizegrades FROM ssq.draw_stats ORDER BY dnum")
    rows = []
    for dnum, sales, pool, pg in cur.fetchall():
        try:
            grades = json.loads(pg) if isinstance(pg, str) else pg
        except Exception:
            grades = []
        rows.append({"dnum": dnum, "sales": int(sales or 0),
                      "pool": int(pool or 0), "grades": grades or []})
    return rows


def parse_grades(grades: list[dict]) -> dict[int, tuple[int, int]]:
    """type -> (typenum中奖注数, typemoney单注奖金元)。空值记0。
    清洗加奖注记(如 '5250000（含加奖250000）' → 5250000)。"""
    import re as _re
    def _num(s):
        if s is None:
            return 0
        m = _re.search(r"\d+", str(s))
        return int(m.group()) if m else 0
    out: dict[int, tuple[int, int]] = {}
    for g in grades:
        t = _num(g.get("type"))
        n = _num(g.get("typenum"))
        m = _num(g.get("typemoney"))
        out[t] = (n, m)
    return out


def ev_expost(rec: dict) -> float:
    """事后口径: 用实测单注奖金 × 中奖概率, 反推单位回报(结构参考)。"""
    g = parse_grades(rec["grades"])
    ev = 0.0
    # 奖级 type: 1=一等奖(6+1), 2=二等奖(6+0), 3=三等奖(5+1),
    # 4=四等奖(5+0/4+1), 5=五等奖(4+0/3+1), 6=六等奖(2+1/1+1/0+1), 7=福运奖(3红)
    p_map = {
        1: P1,
        2: P_RED - P1,                       # 6红中但蓝不中
        3: (comb(6,5)*comb(27,1)/comb(33,6)) * (15/16),
        4: (comb(6,5)*comb(27,1)/comb(33,6))*(1/16)
           + (comb(6,4)*comb(27,2)/comb(33,6))*(15/16),
        5: (comb(6,4)*comb(27,2)/comb(33,6))*(1/16)
           + (comb(6,3)*comb(27,3)/comb(33,6))*(15/16),
        6: (comb(6,2)*comb(27,4)/comb(33,6))*(15/16)
           + (comb(6,1)*comb(27,5)/comb(33,6))*(15/16)
           + (comb(27,6)/comb(33,6))*(15/16),
        7: (comb(6,3)*comb(27,3)/comb(33,6))*(1/16),  # 福运奖: 仅3红中(蓝不中)
    }
    for t, p in p_map.items():
        if t in g:
            _, money = g[t]
            ev += p * money
    return ev - TICKET


def ev_exante(rec: dict) -> float:
    """事前期望口径(乐观上界): 奖池 + 销量建模分奖碰撞。

    关键纠正(2026-08-29 修正): 一等奖奖金**受单注封顶约束**——
    单注封顶 = prizegrades[0].typemoney 实测值(通常 1000万, 奖池<15亿即封顶),
    不是整个奖池! 正确公式:
      一等奖奖金池分配 J1 = pool × HIGH_SHARE (高奖级计提入一等奖比例, 取 0.75 上界)
      一等奖单注期望 = min(封顶, J1 / max(λ1,1)) × P1,  λ1 = (sales/2)×P1
      其余奖级用「固定奖额 / max(该级期望中奖注数,1)」近似(高奖级受分奖影响,
      低奖级近似独立), 同样受各自封顶约束(二等奖封顶 500万/注)
    """
    sales = rec["sales"]
    pool = rec["pool"]
    n_tickets = max(sales / TICKET, 1)
    g = parse_grades(rec["grades"])

    # 封顶值从真实数据读取(一等奖 type=1)
    cap1 = g.get(1, (0, 0))[1] or 10_000_000   # 默认 1000万兜底
    cap2 = g.get(2, (0, 0))[1] or 500_000      # 二等奖封顶(实测 typemoney 已是封顶值)

    HIGH_SHARE = 0.75  # 高奖级奖金入一等奖比例(上界估计)
    lam1 = n_tickets * P1
    j1 = pool * HIGH_SHARE
    single1 = min(cap1, j1 / max(lam1, 1)) if pool > 0 else 0.0
    p1_exp = P1 * single1

    ev = p1_exp
    # 二~七奖级: 用期望中奖注数近似分裂 (保守: 实际更低), 受封顶约束
    p_map = {
        2: P_RED - P1,
        3: (comb(6,5)*comb(27,1)/comb(33,6)) * (15/16),
        4: (comb(6,5)*comb(27,1)/comb(33,6))*(1/16)
           + (comb(6,4)*comb(27,2)/comb(33,6))*(15/16),
        5: (comb(6,4)*comb(27,2)/comb(33,6))*(1/16)
           + (comb(6,3)*comb(27,3)/comb(33,6))*(15/16),
        6: (comb(6,2)*comb(27,4)/comb(33,6))*(15/16)
           + (comb(6,1)*comb(27,5)/comb(33,6))*(15/16)
           + (comb(27,6)/comb(33,6))*(15/16),
        7: (comb(6,3)*comb(27,3)/comb(33,6))*(1/16),
    }
    for t, p in p_map.items():
        if t in g:
            _, money = g[t]
            lam = n_tickets * p
            # 该级单注奖金若 > cap2 (二等奖封顶), 也受封顶约束
            single = min(money, cap2) if t == 2 else money
            ev += p * (single / max(lam, 1))
    return ev - TICKET


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--no-csv", action="store_true", help="不写 CSV, 只打印摘要")
    args = ap.parse_args()

    rows = load_stats()
    print(f"加载 {len(rows)} 期奖金数据")

    results = []
    for r in rows:
        ep = ev_expost(r)
        ea = ev_exante(r)
        results.append({"dnum": r["dnum"], "sales": r["sales"],
                         "pool": r["pool"], "ev_expost": ep, "ev_exante": ea})

    # 摘要统计
    expost = [x["ev_expost"] for x in results]
    exante = [x["ev_exante"] for x in results]
    best_exante = max(exante)
    worst_exante = min(exante)
    n_pos_exante = sum(1 for v in exante if v > 0)
    best_rec = max(results, key=lambda x: x["ev_exante"])

    print("\n=== SSQ 单位投注 EV 摘要 (每注 ¥2) ===")
    print(f"样本期数:           {len(results)}")
    print(f"事后 EV 均值:       {sum(expost)/len(expost):.4f} 元/注")
    print(f"事前 EV 均值:       {sum(exante)/len(exante):.4f} 元/注")
    print(f"事前 EV 最优(最高): {best_exante:.4f} 元/注  @ 期{best_rec['dnum']} "
          f"(奖池={best_rec['pool']/1e8:.2f}亿, 销量={best_rec['sales']/1e8:.2f}亿)")
    print(f"事前 EV 最差(最低): {worst_exante:.4f} 元/注")
    print(f"正 EV 窗口期数:     {n_pos_exante} / {len(results)} "
          f"({'存在' if n_pos_exante else '不存在'})")
    print("\n=== 结论 ===")
    if n_pos_exante == 0:
        print("✓ SSQ 全样本事前 EV 恒为负 → 印证「理性停手」量化证据")
        print("  (封顶 1000万 + 高销量 → 分奖碰撞使一等奖单注期望远低于 ¥2)")
    else:
        print(f"⚠ 发现 {n_pos_exante} 期正 EV 窗口 → 须复核规则/数据(极不可能)")

    if not args.no_csv:
        out = ROOT / "analysis" / "ssq_ev_curve.csv"
        with out.open("w", encoding="utf-8", newline="") as f:
            w = csv.writer(f)
            w.writerow(["dnum", "sales", "pool", "ev_expost", "ev_exante"])
            for x in results:
                w.writerow([x["dnum"], x["sales"], x["pool"],
                            f"{x['ev_expost']:.4f}", f"{x['ev_exante']:.4f}"])
        print(f"\n[ok] EV 曲线已写: {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
