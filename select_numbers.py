#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""从 PostgreSQL 读取最新一次模型预测概率，集成生成 5 组双色球候选号码。

逻辑(完全基于 ml/main.py 已训练模型的输出, 不做额外预测):
  1. 取 PG ssq.model_predictions 中最新 run_at 的 4 个模型概率。
  2. 红球(1-33)/蓝球(1-16) 分别对各模型概率取均值 -> 集成概率。
  3. 红球: 在集成概率上做受控随机加权抽样(softmax 温度), 生成5注,
     每注6个互不相同的号, 并约束奇偶比∈{2:4,3:3,4:2}、大小比(1-16小/17-33大)∈{2:4,3:3,4:2}。
  4. 蓝球: 取集成概率 Top 并结合受控随机, 每注1个。
  5. 输出每注的选取依据(命中哪些模型的高概率号)。

用法:
  .venv/bin/python select_numbers.py                 # 生成5组并打印
  .venv/bin/python select_numbers.py --groups 5 --seed 42
  .venv/bin/python select_numbers.py --wheel         # 旋转矩阵覆盖模式(红球池 Top18)
  .venv/bin/python select_numbers.py --wheel --pool-size 15 --max-notes 30 --no-popularity
"""
from __future__ import annotations
import argparse
import json
import sys
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

import numpy as np
import psycopg

from ml.config import POPULARITY_CONFIG, WHEEL_CONFIG
from ml.popularity import combo_popularity, sample_with_popularity
from wheel import CoverResult, greedy_cover

PG = dict(host="127.0.0.1", port=5432, user="hermes", password="hermes123", dbname="hermes")
SCHEMA = "ssq"
MODELS = ["lstm_blue", "lstm_reds", "lstm_all", "cnn_math"]


def load_latest_probs(conn):
    """读取最新一次 run 的各模型概率, 返回 (red_mean[33], blue_mean[16], run_at, models)。"""
    with conn.cursor() as cur:
        cur.execute(f"""
            SELECT run_at FROM {SCHEMA}.model_predictions
            ORDER BY run_at DESC LIMIT 1
        """)
        row = cur.fetchone()
        if not row:
            raise RuntimeError("PG 中无预测数据, 请先运行 batch_predict_pg.py")
        run_at = row[0]
        cur.execute(f"""
            SELECT model, ball_type, num, prob FROM {SCHEMA}.model_predictions
            WHERE run_at = %s ORDER BY model, ball_type, num
        """, (run_at,))
        red = {}
        blue = {}
        for model, btype, num, prob in cur.fetchall():
            red.setdefault(model, np.zeros(33))
            blue.setdefault(model, np.zeros(16))
            if btype == "red":
                red[model][num - 1] = prob
            elif btype == "blue":
                blue[model][num - 1] = prob
    models = list(red.keys())
    # 集成: 对红/蓝分别按"有该侧数据的模型"求均值(部分输出模型只贡献一侧)
    red_models = [m for m in models if red[m].sum() > 0]
    blue_models = [m for m in models if blue[m].sum() > 0]
    if not red_models or not blue_models:
        raise RuntimeError(f"集成失败: 红球模型={red_models} 蓝球模型={blue_models}")
    red_mean = np.stack([red[m] for m in red_models]).mean(axis=0)
    blue_mean = np.stack([blue[m] for m in blue_models]).mean(axis=0)
    return red_mean, blue_mean, run_at, models


def _sample_red(red_prob, rng, temperature=0.6, popularity_fn=None, lambda_=0.3):
    """按受控随机加权抽样6个不重复红球, 约束奇偶/大小比。

    popularity_fn 不为 None 时: 先按温度采样生成候选注, 再按流行度惩罚加权
    重采样(偏向冷门组合); 为 None 走原逻辑(向后兼容, generate 默认路径不变)。
    """
    if popularity_fn is not None:
        return sample_with_popularity(red_prob, rng, temperature=temperature,
                                      lambda_=lambda_, popularity_fn=popularity_fn)
    for _ in range(200):
        p = np.power(red_prob, 1.0 / temperature)
        p = p / p.sum()
        picks = rng.choice(33, size=6, replace=False, p=p) + 1
        odds = sum(1 for x in picks if x % 2 == 1)
        big = sum(1 for x in picks if x > 16)
        if odds in (2, 3, 4) and big in (2, 3, 4):
            return sorted(picks.tolist())
    # 兜底: 直接取Top6
    return sorted((np.argsort(red_prob)[-6:] + 1).tolist())


def _sample_blue(blue_prob, rng, temperature=0.7):
    p = np.power(blue_prob, 1.0 / temperature)
    p = p / p.sum()
    return int(rng.choice(16, size=1, p=p)[0] + 1)


def generate(red_mean, blue_mean, groups=5, seed=42):
    rng = np.random.default_rng(seed)
    out = []
    for g in range(groups):
        reds = _sample_red(red_mean, rng)
        blue = _sample_blue(blue_mean, rng)
        # 选取依据: 该红球在集成概率中的排名(前8为热号)
        red_rank = {int(n) + 1: int(r) + 1 for r, n in enumerate(np.argsort(red_mean)[::-1])}
        blue_rank = {int(n) + 1: int(r) + 1 for r, n in enumerate(np.argsort(blue_mean)[::-1])}
        hot_reds = [n for n in reds if red_rank[n] <= 8]
        out.append({
            "group": g + 1,
            "red": reds,
            "blue": blue,
            "hot_reds": hot_reds,
            "blue_rank": blue_rank[blue],
        })
    return out


def _popularity_off(reds):
    """--no-popularity 时使用的流行度函数: 不计算流行度(返回 None)。"""
    return None


def build_wheel_tickets(red_mean, blue_mean, pool_size=18, max_notes=30, restarts=3,
                        seed=42, popularity_fn=None, lambda_=0.3):
    """旋转矩阵覆盖模式: 红球 Top-pool_size 概率池 -> 贪心覆盖 -> 每注配 1 个蓝球。

    Args:
        red_mean: 33 维红球集成概率。
        blue_mean: 16 维蓝球集成概率。
        pool_size: 概率 Top-N 红球池大小(6..33)。
        max_notes / restarts / seed: 透传 wheel.greedy_cover。
        popularity_fn: 每注流行度计算函数(reds -> float); None 时用默认
            combo_popularity(规则版)。
        lambda_: 流行度惩罚系数, 保留给惩罚加权路径(与 sample_with_popularity
            同款参数); 默认路径不影响 combo_popularity 输出。

    Returns:
        {"tickets": [{"reds": [...], "blue": n, "popularity": float|None}, ...],
         "coverage": {CoverResult 字段 + pool/pool_size}}
    """
    red_mean = np.asarray(red_mean, dtype=float)
    blue_mean = np.asarray(blue_mean, dtype=float)
    if len(red_mean) != 33 or len(blue_mean) != 16:
        raise ValueError(f"red_mean 长度必须为33, blue_mean 必须为16, "
                         f"实际 {len(red_mean)}/{len(blue_mean)}")
    if not (6 <= pool_size <= 33):
        raise ValueError(f"pool_size 必须在 6-33 之间, 实际 {pool_size}")
    top = np.argsort(red_mean)[::-1][:pool_size]
    pool = sorted((top + 1).tolist())
    res: CoverResult = greedy_cover(pool, k=6, t=4, max_notes=max_notes,
                                    restarts=restarts, seed=seed)
    blue_rng = np.random.default_rng(seed + 1)
    tickets = []
    for reds in res.tickets:
        blue = _sample_blue(blue_mean, blue_rng, temperature=0.7)
        popularity = popularity_fn(reds) if popularity_fn is not None \
            else combo_popularity(reds)
        tickets.append({"reds": reds, "blue": blue, "popularity": popularity})
    coverage = {
        "n_notes": res.n_notes,
        "covered_4subsets": res.covered_4subsets,
        "total_4subsets": res.total_4subsets,
        "four_subset_coverage": res.four_subset_coverage,
        "pass_rate": res.pass_rate,
        "pass_rate_sampled": res.pass_rate_sampled,
        "converged": res.converged,
        "max_notes": res.max_notes,
        "pool": pool,
        "pool_size": len(pool),
    }
    return {"tickets": tickets, "coverage": coverage}


def _run_wheel(red_mean, blue_mean, args, run_at):
    """--wheel 模式: 生成并打印覆盖注单 + 覆盖率报告。"""
    pop_fn = _popularity_off if args.no_popularity else None
    result = build_wheel_tickets(red_mean, blue_mean, pool_size=args.pool_size,
                                 max_notes=args.max_notes, restarts=3, seed=args.seed,
                                 popularity_fn=pop_fn, lambda_=args.popularity_lambda)
    cov = result["coverage"]
    if args.json:
        print(json.dumps({"run_at": str(run_at), "mode": "wheel", **result},
                         ensure_ascii=False, indent=2))
        return
    print(f"模型集成预测 ({run_at}) | 旋转矩阵覆盖模式: 红球池 Top-{cov['pool_size']}")
    print("=" * 56)
    for i, t in enumerate(result["tickets"], 1):
        pop = t["popularity"]
        pop_s = f"{pop:.3f}" if isinstance(pop, float) else "—"
        print(f"注{i:02d}: 红球 {t['reds']} + 蓝球 {t['blue']:02d} | 流行度 {pop_s}")
    print("=" * 56)
    pr_s = f"{cov['pass_rate'] * 100:.2f}%"
    if cov["pass_rate_sampled"] is not None:
        pr_s += " (抽样±0.35%)"
    else:
        pr_s += " (精确)"
    print(f"[覆盖报告] 4-子集覆盖率: {cov['covered_4subsets']}/{cov['total_4subsets']} "
          f"= {cov['four_subset_coverage'] * 100:.2f}%")
    print(f"[覆盖报告] 6-子集通过率: {pr_s} | 注数 {cov['n_notes']}/{cov['max_notes']} "
          f"| 收敛: {cov['converged']} | 池大小: {cov['pool_size']}")
    print("注: 覆盖设计为概率性保证(6-子集通过率≥99% 阈值在池≤16/25注可达), 仅供娱乐参考")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--groups", type=int, default=5)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--json", action="store_true", help="输出 JSON")
    ap.add_argument("--wheel", action="store_true",
                    help="旋转矩阵覆盖模式(概率性保证, 覆盖 Top-N 概率池)")
    ap.add_argument("--pool-size", type=int, default=WHEEL_CONFIG.get("pool_size", 18),
                    help="红球概率池大小(6..33)")
    ap.add_argument("--max-notes", type=int, default=WHEEL_CONFIG.get("max_notes", 30),
                    help="旋转矩阵最大注数")
    ap.add_argument("--no-popularity", action="store_true", help="关闭流行度惩罚")
    ap.add_argument("--popularity-lambda", type=float,
                    default=POPULARITY_CONFIG.get("lambda", 0.3),
                    help="流行度惩罚系数 λ")
    args = ap.parse_args()

    conn = psycopg.connect(**PG)
    try:
        red_mean, blue_mean, run_at, _ = load_latest_probs(conn)
    finally:
        conn.close()

    if args.wheel:
        _run_wheel(red_mean, blue_mean, args, run_at)
        return

    groups = generate(red_mean, blue_mean, args.groups, args.seed)

    if args.json:
        print(json.dumps({"run_at": str(run_at), "groups": groups},
                         ensure_ascii=False, indent=2))
        return

    print(f"模型集成预测 ({run_at}) | 红球Top8: "
          f"{[int(x)+1 for x in np.argsort(red_mean)[::-1][:8]]} "
          f"蓝球Top5: {[int(x)+1 for x in np.argsort(blue_mean)[::-1][:5]]}")
    print("=" * 56)
    for g in groups:
        print(f"第{g['group']}组: 红球 {g['red']} + 蓝球 {g['blue']:02d}")
        print(f"       热号(集成概率前8): 红球 {g['hot_reds']} | 蓝球排名 #{g['blue_rank']}")
    print("=" * 56)
    print("注: 以上基于 ML 模型历史概率集成, 仅供娱乐参考, 不保证中奖。")


if __name__ == "__main__":
    main()
