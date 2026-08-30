#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Top5 与 Wheel(10/20/30) 结合对比: 去重后总注数 / 重叠 / 覆盖率。

只读 PG 最新集成概率, 不落库、不发信。用于回答"wheel10/20/30 结合 Top5 有何不同"。
walk-forward 准确性结论见 compare_top5_wheel.py(四法均=随机下限, 此脚本只比'消化方式'性价比)。
"""
from __future__ import annotations
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent  # python/SSQ
sys.path.insert(0, str(ROOT))

import numpy as np
import psycopg
from select_numbers import load_latest_probs, build_wheel_tickets
from ssq_send_picks import gen_top5
from ml.pg_conn import pg_dict

PG = pg_dict()


def tkey(t: dict) -> tuple:
    """注的唯一键: (红球排序元组, 蓝球)。"""
    return (tuple(sorted(t["reds"])), t["blue"])


def main() -> None:
    period, seed, _ = __import__("ssq_send_picks").compute_next_period()
    conn = psycopg.connect(**PG)
    red_mean, blue_mean, run_at, _ = load_latest_probs(conn)
    conn.close()

    rng = np.random.default_rng(seed)
    top5 = gen_top5(red_mean, blue_mean, rng)  # 生产同款 popularity 冷门加权
    top5_t = [{"reds": g["reds"], "blue": g["blue"]} for g in top5]
    print(f"目标期={period} seed={seed} run_at={run_at}")
    print(f"Top5 注数 = {len(top5_t)}")
    print("=" * 86)
    print(f"{'Wheel':<8}{'wheel注':>8}{'与Top5重叠':>12}{'合并去重总注':>14}"
          f"{'6-子集通过率':>15}{'4-子集覆盖':>12}{'总成本(元)':>12}")
    print("-" * 86)

    rows = []
    for wn in (10, 20, 30):
        res = build_wheel_tickets(red_mean, blue_mean, pool_size=18,
                                  max_notes=wn, restarts=3, seed=seed)
        wt = res["tickets"]
        cov = res["coverage"]
        combined = list({tkey(t): t for t in (top5_t + wt)}.values())
        overlap = len(top5_t) + len(wt) - len(combined)
        cost = len(combined) * 2
        pr = cov["pass_rate"] * 100
        fsc = cov["four_subset_coverage"] * 100
        rows.append((wn, len(wt), overlap, len(combined), pr, fsc, cost))
        print(f"W{wn:<5}{len(wt):>8}{overlap:>12}{len(combined):>14}"
              f"{pr:>14.2f}%{fsc:>11.2f}%{cost:>12}")

    print("=" * 86)
    # 诚实解读
    base_cost = len(top5_t) * 2
    print(f"纯 Top5 成本 = {base_cost} 元 (5注)")
    print(f"原方案 Top5+Wheel30 成本 = {5*2 + 30*2} 元 (35注, 含重叠浪费)")
    print(f"推荐  Top5+Wheel20 去重后 ≈ {[r[3] for r in rows if r[0]==20][0]} 注 "
          f"≈ {[r[6] for r in rows if r[0]==20][0]} 元")
    print("\n注: 四种方法吃同一份 FLAT 概率, 每注命中率均=随机下限(1.09红/注);")
    print("    差异只在'覆盖规模/成本', 不在'精度'。去重消除的是重复花钱买同一组合。")


if __name__ == "__main__":
    main()
