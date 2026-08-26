#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""旋转矩阵(wheel) ROI 模拟账本 —— 诚实的盈亏核算。

目的(Rocky 指示 2026-08-13): 基于旋转矩阵 wheel, 假设每期买入 30 注(每注 1 份/2元),
对全部历史开奖逐期核算盈亏, 量化"长期到底亏多少、wheel 相比纯随机是否更省"。

口径(已在 2026-08-13 与 Rocky 对齐):
  B. 固定快照: 用 08-13 那次 6 模型集成概率(red_mean)选 Top-pool_size 红球做池子,
     生成一份固定的 30 注 wheel + 一份固定的 30 注纯随机, 拿这两份去比*所有*历史开奖。
  C. 纯随机对照: 30 注完全随机, 同样比全部历史, 作为 baseline。

关键诚实声明: 双色球为独立均匀随机过程, 任何选号策略的期望收益 < 成本。
本账本不证明"能赚钱", 而是坐实"长期必亏"并用数据说话(反向思维/诚实检验)。

奖级与单注奖金(简化, 取常见固定值, 用于模拟):
  6+1 一等奖 ~5,000,000 ; 6+0 二等奖 ~100,000 ; 5+1 三等奖 ~3,000 ;
  5+0 / 4+1 四等奖 ~200 ; 4+0 / 3+1 五等奖 ~10 ; 2+1 / 1+1 / 0+1 六等奖 ~5

用法:
  .venv/bin/python analysis/wheel_ledger.py
输出: wheel 与 random 两份账本的逐期盈亏汇总 + 中奖分布 + 期望对比。
"""
from __future__ import annotations

import csv
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from ml.config import WHEEL_CONFIG  # noqa: E402
from wheel import greedy_cover  # noqa: E402
from select_numbers import load_latest_probs  # noqa: E402

CSV_PATH = ROOT / "ml" / "data" / "1.csv"
COST_PER_NOTE = 2.0  # 每注 2 元

# 奖级单注奖金(模拟用固定值)
PRIZE = {
    (6, 1): 5_000_000, (6, 0): 100_000, (5, 1): 3_000,
    (5, 0): 200, (4, 1): 200, (4, 0): 10, (3, 1): 10,
    (2, 1): 5, (1, 1): 5, (0, 1): 5,
}


def load_history(path: Path):
    """读取 1.csv, 返回 [(issue, set(reds), blue), ...] 升序。"""
    rows = []
    with open(path, encoding="utf-8-sig", newline="") as f:
        r = csv.DictReader(f)
        for row in r:
            reds = tuple(int(row[f"Red{i}"]) for i in range(1, 7))
            blue = int(row["Blue1"])
            rows.append((row["dNum"], frozenset(reds), blue))
    return rows


def winnings(red_match: int, blue_match: int) -> float:
    return float(PRIZE.get((red_match, blue_match), 0))


def build_fixed_wheel(red_mean: np.ndarray, pool_size: int, max_notes: int, seed: int):
    """用 red_mean 选 Top-pool_size 池子 -> 固定 30 注 wheel(红球)。"""
    top_idx = np.argsort(red_mean)[-pool_size:] + 1  # 1..33
    pool = sorted(int(x) for x in top_idx)
    res = greedy_cover(pool, max_notes=max_notes, restarts=3, seed=seed)
    return res.tickets  # list[list[int]]


def build_fixed_random(n_notes: int, seed: int):
    """固定 30 注纯随机(可复现)。"""
    rng = np.random.default_rng(seed)
    tickets = []
    while len(tickets) < n_notes:
        t = tuple(sorted(rng.choice(33, size=6, replace=False) + 1))
        if t not in tickets:
            tickets.append(t)
    return [list(t) for t in tickets]


def simulate_ledger(tickets: list, history: list, label: str) -> dict:
    """对全部历史开奖, 逐期算这 30 注的盈亏。

    口径说明: 本账本衡量"红球 wheel 覆盖率"的价值, 蓝球不配
    (wheel 模式蓝球独立出号, 其命中属独立随机, 不计入 wheel 贡献)。
    因此每注只按"红球命中数"计奖级, 蓝球命中计 0。
    """
    n_notes = len(tickets)
    ticket_sets = [frozenset(t) for t in tickets]
    total_cost = COST_PER_NOTE * n_notes * len(history)
    total_win = 0.0
    best_red = 0
    prize_dist: dict = {}
    winning_draws = 0
    for _, draw_reds, _draw_blue in history:
        for ts in ticket_sets:
            rm = len(ts & draw_reds)
            w = winnings(rm, 0)  # 蓝球不配, 计 0
            if w > 0:
                prize_dist[(rm, 0)] = prize_dist.get((rm, 0), 0) + 1
                winning_draws += 1
            total_win += w
            best_red = max(best_red, rm)
    net = total_win - total_cost
    roi = (net / total_cost) if total_cost else 0.0
    return {
        "label": label,
        "n_notes": n_notes,
        "n_periods": len(history),
        "total_cost": total_cost,
        "total_win": total_win,
        "net": net,
        "roi": roi,
        "best_red_match": best_red,
        "prize_dist": dict(sorted(prize_dist.items())),
        "winning_draws": winning_draws,
    }


def main():
    print("=" * 64)
    print("旋转矩阵(wheel) ROI 模拟账本  | 口径 B+C (固定快照 + 随机对照)")
    print("=" * 64)

    history = load_history(CSV_PATH)
    print(f"历史开奖期数: {len(history)} (从 {history[0][0]} 到 {history[-1][0]})")

    # 加载真实模型概率(08-13 run) 作为 wheel 池子来源
    import psycopg
    from ml.pg_conn import pg_dict
    PG = pg_dict()  # 凭证从 ~/.hermes/.env 的 DATABASE_URL 读, 不硬编码
    with psycopg.connect(**PG) as conn:
        red_mean, blue_mean, run_at, models = load_latest_probs(conn)
    print(f"模型概率来源: run_at={run_at}, 模型={models}")
    print(f"Wheel 配置: pool_size={WHEEL_CONFIG['pool_size']}, "
          f"max_notes={WHEEL_CONFIG['max_notes']}")

    pool_size = WHEEL_CONFIG["pool_size"]
    max_notes = WHEEL_CONFIG["max_notes"]

    # B. 固定 wheel (用真实 red_mean 选池)
    wheel_tickets = build_fixed_wheel(red_mean, pool_size, max_notes, seed=20260813)
    # C. 固定随机对照
    rand_tickets = build_fixed_random(max_notes, seed=20260813)

    print(f"\n生成 wheel {len(wheel_tickets)} 注 / 随机 {len(rand_tickets)} 注 (可复现 seed=20260813)")

    wheel_ledger = simulate_ledger(wheel_tickets, history, "WHEEL(旋转矩阵)")
    rand_ledger = simulate_ledger(rand_tickets, history, "RANDOM(纯随机)")

    for L in (wheel_ledger, rand_ledger):
        print("\n" + "-" * 64)
        print(f"【{L['label']}】 每期 {L['n_notes']} 注 × {L['n_periods']} 期")
        print(f"  总投入 : {L['total_cost']:,.0f} 元")
        print(f"  总奖金 : {L['total_win']:,.0f} 元")
        print(f"  净盈亏 : {L['net']:,.0f} 元")
        print(f"  ROI    : {L['roi']*100:,.2f}%")
        print(f"  历史最佳红球命中: {L['best_red_match']} 红")
        print(f"  中奖期数(任一注中 >=4红 或 奖级>0): {L['winning_draws']}")
        if L["prize_dist"]:
            print("  奖级分布 (红球命中,蓝球命中 -> 次数):")
            for (rm, bm), c in L["prize_dist"].items():
                print(f"    ({rm},{bm}) -> {c} 次")

    print("\n" + "=" * 64)
    print("结论(诚实): 期望收益 < 成本, 长期必亏。wheel 的价值是")
    print("'中4+红覆盖率'更稳(小奖更密), 而非提升头奖概率。")
    delta = wheel_ledger["net"] - rand_ledger["net"]
    print(f"wheel 相对 random 净盈亏差: {delta:,.0f} 元 "
          f"({'wheel更省' if delta > 0 else 'random更省'})")
    print("=" * 64)


if __name__ == "__main__":
    main()
