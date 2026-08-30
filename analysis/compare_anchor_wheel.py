#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""完整锚策略 × Wheel 结合对比: popularity vs pareto × N=10/20/30。

对比维度 (每组 R 次重启取均值±std, 消除单次随机波动):
  1. 结合总注数 (应恒 = N, 验证"结合非叠加")
  2. 6-子集通过率 pass_rate (来自 wheel 覆盖, 与锚无关 → 两法应一致)
  3. 4-子集覆盖率 four_subset_coverage (同上)
  4. Top5 锚三目标: f_prob / f_cool / f_spread (越大越好)
  5. 锚平均流行度 avg_popularity (越低越避撞)
  6. 锚去重红球数 / 结合去重红球数 (覆盖广度)
  7. 成本 = N × 2 元

诚实边界: 命中率两法均 FLAT (随机下限, 见 compare_top5_wheel.py walk-forward)。
本对比只评"选号质量/收益维度": pareto 应更避撞(f_cool↑/pop↓)、更分散(f_spread↑)、
覆盖更广(combined_distinct↑)。pass_rate 两法一致 (锚不进 wheel 池)。

用法: cd /home/hermes/workspace/python/SSQ && .venv/bin/python analysis/compare_anchor_wheel.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from ml.pg_conn import connect
import select_numbers as sn
from ml.popularity import combo_popularity
from ml.pareto_select import score_ticket, set_distinct_reds, set_objective_means
from ssq_send_picks import gen_top5, gen_top5_pareto, gen_wheel, merge_top5_wheel


def anchor_scores(notes, red_mean, blue_mean):
    """给 legacy 锚补算三目标 (pareto 锚自带 score)。"""
    out = []
    for t in notes:
        sc = t.get("score") or score_ticket(t["reds"], t["blue"], red_mean, blue_mean)
        out.append({"reds": t["reds"], "blue": t["blue"],
                    "popularity": float(combo_popularity(t["reds"])),
                    "score": sc})
    return out


def trial(anchor_mode, N, red_mean, blue_mean, seed):
    rng = np.random.default_rng(seed)
    if anchor_mode == "pareto":
        top5 = gen_top5_pareto(red_mean, blue_mean, rng, pool_size=300, top5_count=5)
    else:
        top5 = gen_top5(red_mean, blue_mean, rng)
    top5 = anchor_scores(top5, red_mean, blue_mean)
    wheel = gen_wheel(red_mean, blue_mean, seed, total_notes=N, extra=5)
    merged = merge_top5_wheel(top5, wheel["tickets"], N)
    cov = wheel["coverage"]
    am = set_objective_means(top5)
    return {
        "total": len(merged),
        "pass_rate": cov.get("pass_rate", float("nan")),
        "four_cov": cov.get("four_subset_coverage", float("nan")),
        "f_prob": am["f_prob"], "f_cool": am["f_cool"], "f_spread": am["f_spread"],
        "avg_pop": float(np.mean([t["popularity"] for t in top5])),
        "anchor_distinct": set_distinct_reds(top5),
        "combined_distinct": set_distinct_reds(merged),
        "cost": N * 2,
    }


def main():
    conn = connect()
    red_mean, blue_mean, run_at, _ = sn.load_latest_probs(conn)
    conn.close()
    print(f"[data] run_at={run_at}")

    R = 30          # 重启次数
    Ns = [10, 20, 30]
    modes = ["popularity", "pareto"]
    rows = {(m, N): [] for m in modes for N in Ns}

    for i in range(R):
        for N in Ns:
            for m in modes:
                rows[(m, N)].append(trial(m, N, red_mean, blue_mean, 2000 + i))

    def agg(vals, key):
        arr = np.array([v[key] for v in vals], dtype=float)
        return arr.mean(), arr.std()

    for N in Ns:
        print(f"\n=== N={N} (结合总注={N}, 成本{N*2}元, R={R}) ===")
        print(f"{'指标':<22}{'popularity':>14}{'pareto':>14}{'Δ(pop-leg)':>14}")
        print("-" * 66)
        keys = [("total", "结合总注"), ("pass_rate", "6-子集通过率"),
                ("four_cov", "4-子集覆盖率"), ("f_prob", "锚·对齐概率"),
                ("f_cool", "锚·冷门度"), ("f_spread", "锚·数字跨度"),
                ("avg_pop", "锚·平均流行度(低好)"), ("anchor_distinct", "锚·去重红球"),
                ("combined_distinct", "结合·去重红球")]
        for key, label in keys:
            lp, ls = agg(rows[("popularity", N)], key)
            pp, ps = agg(rows[("pareto", N)], key)
            if key in ("pass_rate", "four_cov", "f_prob", "f_cool", "f_spread", "avg_pop"):
                print(f"{label:<20}{lp:>10.4f}±{ls:.3f}{pp:>10.4f}±{ps:.3f}{pp-lp:>+12.4f}")
            else:
                print(f"{label:<20}{lp:>10.2f}±{ls:.2f}{pp:>10.2f}±{ps:.2f}{pp-lp:>+12.2f}")

    print("\n=== 诚实判读 ===")
    print("• pass_rate / four_cov 两法一致 → 锚不进 wheel 池, 覆盖由 wheel 决定")
    print("• pareto 在 f_cool/avg_pop/f_spread 应全面占优 → 更避撞、更分散")
    print("• combined_distinct: pareto 锚把覆盖集撑大 → 结合后去重红球更多")
    print("• 命中率两法均 FLAT (随机下限); 本测试只评选号质量/收益维度")


if __name__ == "__main__":
    main()
