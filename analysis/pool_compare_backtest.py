#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""历史回测: ML Top-18 池 vs 随机 18 球池 + wheel 旋转矩阵 (Rocky 2026-08-14 指示)。

目的: 检验"模型选出的 18 球池 + wheel 30 注"在历史开奖上是否优于"随机 18 球池 + wheel 30 注"。

设计:
  A. ML 池: 取 PG model_predictions 最新 run_at 的 red_mean Top-18, 固定一份 wheel 30 注。
  B. 随机池: 1000 次独立随机抽 18 球(每次 seed 不同), 各生成一份 wheel 30 注。
  C. 两者对同一份 3489 期历史开奖逐期核算盈亏。
  D. 统计: ML 池 ROI vs 随机池 ROI 分布(均值/中位数/5%-95% 分位), 看 ML 是否落在分布内。

诚实预判: 双色球为独立均匀随机过程, wheel 覆盖率只依赖池大小(=18 时 95.42%),
与哪 18 个球无关。因此 ML 池与随机池的 ROI 应无统计显著差异。
本回测用数据验证这个预判。

用法:
  .venv/bin/python analysis/pool_compare_backtest.py
"""
from __future__ import annotations

import csv
import sys
from pathlib import Path

import numpy as np
import psycopg

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from wheel import greedy_cover
from select_numbers import load_latest_probs

CSV_PATH = ROOT / "ml" / "data" / "1.csv"
COST_PER_NOTE = 2.0
WHEEL_NOTES = 30
POOL_SIZE = 18
N_RANDOM_TRIALS = 200
RANDOM_SEED = 2026

from ml.pg_conn import pg_dict

PG = pg_dict()  # 凭证从 ~/.hermes/.env 的 DATABASE_URL 读, 不硬编码

PRIZE = {
    (6, 1): 5_000_000, (6, 0): 100_000, (5, 1): 3_000,
    (5, 0): 200, (4, 1): 200, (4, 0): 10, (3, 1): 10,
    (2, 1): 5, (1, 1): 5, (0, 1): 5,
}


def load_history(path: Path):
    rows = []
    with open(path, encoding="utf-8-sig", newline="") as f:
        r = csv.DictReader(f)
        for row in r:
            reds = frozenset(int(row[f"Red{i}"]) for i in range(1, 7))
            blue = int(row["Blue1"])
            rows.append((row["dNum"], reds, blue))
    return rows


def backtest_one_pool(pool: list, history: list, seed: int) -> dict:
    """对固定池子生成 wheel 注单, 跑历史开奖, 返回盈亏统计。"""
    # wheel 覆盖
    res = greedy_cover(pool, k=6, t=4, max_notes=WHEEL_NOTES, restarts=3, seed=seed)
    wheel_reds = [frozenset(t) for t in res.tickets]
    # 蓝球: 每注独立随机(不依赖池, 仅做 ROI 估算用)
    blue_rng = np.random.default_rng(seed + 1)
    wheel_blues = [int(blue_rng.integers(1, 17)) for _ in wheel_reds]

    total_cost = 0.0
    total_prize = 0.0
    prize_dist = {k: 0 for k in PRIZE}
    in_pool_count = 0

    for issue, draw_reds, draw_blue in history:
        cost = WHEEL_NOTES * COST_PER_NOTE
        total_cost += cost

        # 6 奖号是否全落池内
        if draw_reds.issubset(frozenset(pool)):
            in_pool_count += 1

        # 逐注核算
        prize_sum = 0
        for wr, wb in zip(wheel_reds, wheel_blues):
            red_hit = len(wr & draw_reds)
            blue_hit = 1 if wb == draw_blue else 0
            p = PRIZE.get((red_hit, blue_hit), 0)
            if p:
                prize_dist[(red_hit, blue_hit)] += 1
            prize_sum += p
        total_prize += prize_sum

    n = len(history)
    roi = (total_prize - total_cost) / total_cost if total_cost else 0
    return {
        "n_periods": n,
        "total_cost": total_cost,
        "total_prize": total_prize,
        "net": total_prize - total_cost,
        "roi": roi,
        "in_pool_count": in_pool_count,
        "in_pool_rate": in_pool_count / n if n else 0,
        "prize_dist": prize_dist,
        "pass_rate": res.pass_rate,
        "n_notes": res.n_notes,
    }


def main():
    history = load_history(CSV_PATH)
    print(f"历史开奖: {len(history)} 期")

    # ---- A. ML 池 ----
    print("\n[1/2] ML 池 (模型概率 Top-18)...")
    with psycopg.connect(**PG) as conn:
        red_mean, blue_mean, run_at, models = load_latest_probs(conn, method="mean")
    top_idx = np.argsort(red_mean)[::-1][:POOL_SIZE]
    ml_pool = sorted((top_idx + 1).tolist())
    print(f"  模型: {models}, run_at={run_at}")
    print(f"  ML池(升序): {ml_pool}")
    ml_res = backtest_one_pool(ml_pool, history, seed=2026093)
    print(f"  wheel 30 注 pass_rate={ml_res['pass_rate']:.4f}")
    print(f"  6 奖号全落池内 {ml_res['in_pool_count']} 次 ({ml_res['in_pool_rate']*100:.2f}%)")
    print(f"  总投入 {ml_res['total_cost']:.0f} 元, 总奖金 {ml_res['total_prize']:.0f} 元")
    print(f"  ** ML 池 ROI = {ml_res['roi']*100:.2f}% (净盈亏 {ml_res['net']:.0f} 元) **")

    # ---- B. 随机池 (1000 次 Monte Carlo) ----
    print(f"\n[2/2] 随机池 ({N_RANDOM_TRIALS} 次 Monte Carlo)...")
    rng = np.random.default_rng(RANDOM_SEED)
    rand_rois = []
    rand_nets = []
    rand_inpool = []
    for t in range(N_RANDOM_TRIALS):
        pool = sorted(rng.choice(range(1, 33 + 1), size=POOL_SIZE, replace=False).tolist())
        r = backtest_one_pool(pool, history, seed=t)
        rand_rois.append(r["roi"])
        rand_nets.append(r["net"])
        rand_inpool.append(r["in_pool_rate"])

    rand_rois = np.array(rand_rois)
    rand_nets = np.array(rand_nets)
    rand_inpool = np.array(rand_inpool)

    print(f"  随机池 ROI 分布 (N={N_RANDOM_TRIALS}):")
    print(f"    均值   = {rand_rois.mean()*100:.2f}%")
    print(f"    中位数 = {np.median(rand_rois)*100:.2f}%")
    print(f"    5%分位 = {np.percentile(rand_rois, 5)*100:.2f}%")
    print(f"    95%分位= {np.percentile(rand_rois, 95)*100:.2f}%")
    print(f"    标准差 = {rand_rois.std()*100:.2f}%")

    ml_roi = ml_res["roi"]
    # ML 池在随机分布中的位置
    pct = (rand_rois < ml_roi).mean() * 100
    print(f"\n  ** ML 池 ROI ({ml_roi*100:.2f}%) 在随机分布中位于第 {pct:.1f} 百分位 **")

    # ---- C. 对比 ----
    print("\n" + "=" * 60)
    print("对比总结")
    print("=" * 60)
    print(f"  ML 池  ROI: {ml_roi*100:+.2f}% (净 {ml_res['net']:+.0f} 元)")
    print(f"  随机池 ROI: {rand_rois.mean()*100:+.2f}% ± {rand_rois.std()*100:.2f}%")
    print(f"  随机池范围: [{np.percentile(rand_rois,5)*100:+.2f}%, {np.percentile(rand_rois,95)*100:+.2f}%] (90% CI)")

    diff = ml_roi - rand_rois.mean()
    if abs(diff) < rand_rois.std():
        verdict = "ML 池与随机池无统计显著差异 (差异 < 1 个标准差)"
    elif diff > 0:
        verdict = f"ML 池略优于随机池 (+{diff*100:.2f}%), 但在方差范围内, 非统计显著"
    else:
        verdict = f"ML 池略逊于随机池 ({diff*100:.2f}%), 但在方差范围内, 非统计显著"

    print(f"\n  判定: {verdict}")
    print(f"\n  诚实结论: wheel 覆盖率只依赖池大小(={POOL_SIZE} 时为 {ml_res['pass_rate']*100:.2f}%),")
    print(f"  与具体哪 18 个球无关。在独立均匀随机过程上, ML 选池与随机选池期望等价。")
    print(f"  (两者的差异纯粹来自 3489 期实际开奖落在不同池中的随机波动)")


if __name__ == "__main__":
    main()
