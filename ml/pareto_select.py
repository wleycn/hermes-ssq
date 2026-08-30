#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""轻量 Pareto 多目标选号 (NSGA-II 思路的 10% 实现版)。

设计动机 (研究简报 2026-08-30 [4]):
  现有 gen_top5 用 sample_with_popularity(lambda_=0.3) —— 三个目标
  (概率加权 / 降分奖冷门度 / 数字散布覆盖) 被硬编码权重拍成一个标量,
  权重怎么定从未用数据验证。本模块把"取舍"显式化:
    让一批候选注在三维目标上做**非支配排序**, 返回位于 Pareto 前沿
    (互不碾压) 的代表性锚注 —— 把决策权交还给用户, 而非写死 lambda。

三目标 (全部最大化, 归一化到 [0,1]):
  f_prob   : 与集成模型概率的对齐度 (越高=越贴合模型高概率号)
  f_cool   : 冷门度 = 1 - combo_popularity (越高=越避开连号/生日/全奇偶等热门)
  f_spread : 数字散布度 = (max-min)/32 (越高=号码跨度越大, 覆盖越广, 作覆盖代理)

诚实边界: i.i.d. 下三个目标**都不影响命中率**(随机下限)。Pareto 改善的是
"选号质量/收益维度"(万一中了少分摊、覆盖更合理), 不是预测精度。本模块
只替换 Top5 锚的生成策略, walk-forward 已证命中率 FLAT。

依赖: numpy; select_numbers._sample_red (受控随机+奇偶/大小约束);
      ml.popularity.combo_popularity (冷门度评分)。
"""
from __future__ import annotations

from typing import Dict, List, Sequence

import numpy as np

from ml.popularity import combo_popularity
import select_numbers as sn

OBJECTIVES = ("f_prob", "f_cool", "f_spread")


def score_ticket(reds: Sequence[int], blue: int,
                 red_mean: np.ndarray, blue_mean: np.ndarray) -> Dict[str, float]:
    """单注三维目标打分 (全部 [0,1], 全部越大越好)。"""
    reds_arr = np.asarray(list(reds), dtype=int)
    # f_prob: 对齐模型概率, 用理论上限归一化
    prob = float(red_mean[reds_arr - 1].sum() + blue_mean[blue - 1])
    top6 = float(np.sort(red_mean)[-6:].sum() + blue_mean.max())
    f_prob = prob / top6 if top6 > 0 else 0.0
    # f_cool: 冷门度 = 1 - 组合流行度
    f_cool = 1.0 - float(combo_popularity([int(r) for r in reds_arr]))
    # f_spread: 数字跨度代理覆盖 (1..33 => 最大跨度 32)
    f_spread = (float(reds_arr.max()) - float(reds_arr.min())) / 32.0
    return {"f_prob": f_prob, "f_cool": f_cool, "f_spread": f_spread}


def _dominates(a: Dict[str, float], b: Dict[str, float]) -> bool:
    """a 支配 b: 三维均 >= 且至少一维 > (最大化语义)。"""
    return (all(a[k] >= b[k] for k in OBJECTIVES)
            and any(a[k] > b[k] for k in OBJECTIVES))


def non_dominated_front(tickets: List[dict]) -> List[dict]:
    """非支配排序: 返回 Pareto 前沿 (不被任何其他候选支配的注)。O(n^2)。"""
    front: List[dict] = []
    for i, ti in enumerate(tickets):
        dominated = False
        for j, tj in enumerate(tickets):
            if i == j:
                continue
            if _dominates(tj["score"], ti["score"]):
                dominated = True
                break
        if not dominated:
            front.append(ti)
    return front


def select_representatives(front: List[dict], top5_count: int = 5) -> List[dict]:
    """从前沿挑 top5_count 个代表性锚注: 先保证三维各自极值, 再按均衡分补足。"""
    if len(front) <= top5_count:
        return list(front)
    picks: List[dict] = []
    # 1) 保证每个目标的最优点都在 (互不碾压的三极)
    for key in OBJECTIVES:
        best = max(front, key=lambda t: t["score"][key])
        if best not in picks:
            picks.append(best)
    # 2) 剩余按均衡分(三维和)降序贪心补足, 直到 top5_count
    rest = [t for t in front if t not in picks]
    rest.sort(key=lambda t: sum(t["score"].values()), reverse=True)
    for t in rest:
        if len(picks) >= top5_count:
            break
        picks.append(t)
    return picks[:top5_count]


def build_candidate_pool(red_mean: np.ndarray, blue_mean: np.ndarray,
                         rng: np.random.Generator, n: int = 300) -> List[dict]:
    """生成候选注池: 走 select_numbers._sample_red (温度采样+奇偶/大小约束)。"""
    pool: List[dict] = []
    seen = set()
    attempts = 0
    while len(pool) < n and attempts < n * 5:
        attempts += 1
        reds = sn._sample_red(red_mean, rng)
        blue = int(sn._sample_blue(blue_mean, rng))
        key = (tuple(reds), blue)
        if key in seen:
            continue
        seen.add(key)
        pool.append({"reds": [int(r) for r in reds], "blue": blue,
                     "score": score_ticket(reds, blue, red_mean, blue_mean)})
    return pool


def gen_top5_pareto(red_mean: np.ndarray, blue_mean: np.ndarray,
                    rng: np.random.Generator, pool_size: int = 300,
                    top5_count: int = 5) -> List[dict]:
    """轻量 Pareto 版 Top5 锚: 候选池 -> 非支配排序 -> 代表性锚注。

    返回带 score + popularity 的注单列表 (长度=top5_count), 与 gen_top5 同 schema
    (额外带 'score' 字段供复盘/对比)。
    """
    pool = build_candidate_pool(red_mean, blue_mean, rng, n=pool_size)
    front = non_dominated_front(pool)
    reps = select_representatives(front, top5_count=top5_count)
    for t in reps:
        t["popularity"] = float(combo_popularity(t["reds"]))
    return reps


def set_objective_means(tickets: List[dict]) -> Dict[str, float]:
    """一组注的三目标均值 (用于方法对比)。"""
    if not tickets:
        return {k: 0.0 for k in OBJECTIVES}
    return {k: float(np.mean([t["score"][k] for t in tickets])) for k in OBJECTIVES}


def set_distinct_reds(tickets: List[dict]) -> int:
    """一组注覆盖的不同红球数 (覆盖代理指标)。"""
    s = set()
    for t in tickets:
        s.update(t["reds"])
    return len(s)
