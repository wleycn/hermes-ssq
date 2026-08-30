#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""旧(加权采样) vs 新(轻量 Pareto) Top5 锚 同口径对比。

对比维度 (三组各跑 RNG 重复 N 次取均值, 消除单次随机波动):
  1. 三目标均值 f_prob / f_cool / f_spread (越大越好)
  2. 覆盖指标: 5 注涉及的去重红球数 (set_distinct_reds)
  3. 冷门度分布: 平均 popularity (越低越冷门)
  4. 帕累托前沿规模: 候选池里非支配点占比 (新法独有, 反映'取舍空间')

诚实边界: 两法命中率同为随机下限 (FLAT); 本对比只评"选号质量/收益维度",
          不评预测精度。差异在覆盖工程与避撞, 不在命中。

用法: cd /home/hermes/workspace/python/SSQ && .venv/bin/python analysis/compare_pareto.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import select_numbers as sn
from ml.popularity import combo_popularity, sample_with_popularity
from ml.pareto_select import (
    build_candidate_pool, gen_top5_pareto, non_dominated_front,
    set_distinct_reds, set_objective_means, OBJECTIVES,
)
from ml.pg_conn import connect


def gen_top5_legacy(red_mean, blue_mean, rng):
    """原加权采样 (lambda=0.3) Top5 锚, 同 schema 带 score 供对比。"""
    out = []
    for g in range(5):
        reds = sample_with_popularity(red_mean, rng, temperature=0.6,
                                      lambda_=0.3, n_candidates=200)
        blue = int(sn._sample_blue(blue_mean, rng))
        reds = [int(x) for x in reds]
        out.append({"group": g + 1, "reds": reds, "blue": blue,
                    "popularity": float(combo_popularity(reds)),
                    "score": None})
    # score 补算 (复用 pareto 打分函数)
    for t in out:
        from ml.pareto_select import score_ticket
        t["score"] = score_ticket(t["reds"], t["blue"], red_mean, blue_mean)
    return out


def summarize(tickets):
    """一组注的对比摘要。"""
    means = set_objective_means(tickets)
    avg_pop = float(np.mean([t["popularity"] for t in tickets]))
    distinct = set_distinct_reds(tickets)
    return {"f_prob": means["f_prob"], "f_cool": means["f_cool"],
            "f_spread": means["f_spread"], "avg_popularity": avg_pop,
            "distinct_reds": distinct}


def main():
    conn = connect()
    red_mean, blue_mean, run_at, _models = sn.load_latest_probs(conn)
    conn.close()
    print(f"[data] run_at={run_at} 红均维={red_mean.shape} 蓝均维={blue_mean.shape}")

    N = 30  # 重复次数, 消除单次波动
    legacy_rows, pareto_rows = [], []
    frontier_sizes = []
    for i in range(N):
        seed = 1000 + i
        r1 = np.random.default_rng(seed)
        r2 = np.random.default_rng(seed)
        legacy = gen_top5_legacy(red_mean, blue_mean, r1)
        pareto = gen_top5_pareto(red_mean, blue_mean, r2, pool_size=300, top5_count=5)
        legacy_rows.append(summarize(legacy))
        pareto_rows.append(summarize(pareto))
        # 候选池前沿规模 (新法独有指标)
        pool = build_candidate_pool(red_mean, blue_mean, r2, n=300)
        front = non_dominated_front(pool)
        frontier_sizes.append(len(front) / len(pool))

    def agg(rows):
        return {k: float(np.mean([r[k] for r in rows])) for k in rows[0]}

    L, P = agg(legacy_rows), agg(pareto_rows)
    print(f"\n=== Top5 锚对比 (N={N} 次均值, 全部越大越好) ===")
    print(f"{'指标':<18}{'旧 加权':>12}{'新 Pareto':>12}{'Δ':>10}")
    print("-" * 54)
    metrics = [("f_prob", "对齐概率"), ("f_cool", "冷门度"),
               ("f_spread", "数字跨度"), ("avg_popularity", "平均流行度(低好)"),
               ("distinct_reds", "去重红球数")]
    for key, label in metrics:
        delta = P[key] - L[key]
        print(f"{label:<16}{L[key]:>12.4f}{P[key]:>12.4f}{delta:>+10.4f}")
    print("-" * 54)
    print(f"Pareto 候选池前沿占比: {np.mean(frontier_sizes)*100:.1f}% "
          f"(非支配点占比, 反映取舍空间大小)")

    # 诚实判读
    print("\n=== 判读 ===")
    print("两法命中率均为随机下限 (FLAT), 本对比只评'选号质量/收益维度'。")
    print(f"- f_cool/avg_popularity: 新法若更高 → 更避撞(少分摊)")
    print(f"- f_spread/distinct_reds: 新法若更高 → 覆盖更广")
    print(f"- f_prob: 两法应接近 (都贴合模型概率, 差异小)")


if __name__ == "__main__":
    main()
