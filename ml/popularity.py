#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""双色球组合流行度: 6 条规则加权评分 + 冷门组合加权采样。

用途: 识别"可能被大量彩民选中的热门组合"(全奇/连号/生日号/同尾等), 在采样时
按流行度反向加权(penalty), 偏向冷门组合 —— 降低与他人撞号、奖金被分摊的概率。

规则集(每条规则输出归一化 0..1, 综合分 = Σw_i·rule_i / Σw_i):
  1. consecutive_pairs     连号对数 / 5                        权重 1.0
  2. arithmetic_progressions 等差子序列计数(步长1..6)/6, 低起点(≤10)×1.5  权重 1.5
  3. birthday_ratio        生日号(≤31)占比 / 6                 权重 0.8
  4. all_same_parity       全奇或全偶=1 否则 0                  权重 2.0
  5. same_tail_pairs       同尾数对计数 / C(6,2)                权重 0.8
  6. lucky_taboo           (count(6,8,9) − count(4)) / 6       权重 1.2

依赖: numpy(与 select_numbers 一致), 候选注生成逻辑复用 select_numbers._sample_red
的温度采样 + 奇偶/大小比约束。
"""
from __future__ import annotations

from collections import Counter
from typing import Callable, List, Optional, Sequence

import numpy as np

# 默认规则权重(与 ml/config.py POPULARITY_CONFIG["weights"] 保持一致)
DEFAULT_WEIGHTS = {
    "consecutive_pairs": 1.0,
    "arithmetic_progressions": 1.5,
    "birthday_ratio": 0.8,
    "all_same_parity": 2.0,
    "same_tail_pairs": 0.8,
    "lucky_taboo": 1.2,
}


def _clamp01(x: float) -> float:
    return min(1.0, max(0.0, x))


# ---------------- 6 条规则(每条输出归一化 0..1) ----------------

def _rule_consecutive_pairs(reds: Sequence[int]) -> float:
    """连号对数 / 5: 相邻数对 {i, i+1} 计数(6 个数最多 5 对)。"""
    s = set(reds)
    cnt = sum(1 for x in s if (x + 1) in s)
    return cnt / 5.0


def _rule_arithmetic_progressions(reds: Sequence[int]) -> float:
    """等差子序列计数(≥3 项, 步长 1..6) / 6; 存在低起点(≤10)序列再 ×1.5。"""
    s = set(reds)
    cnt = 0
    low_start = False
    for d in range(1, 7):
        for a in s:
            if (a + d) in s and (a + 2 * d) in s:
                cnt += 1
                if a <= 10:
                    low_start = True
    val = cnt / 6.0
    if low_start:
        val *= 1.5
    return _clamp01(val)


def _rule_birthday_ratio(reds: Sequence[int]) -> float:
    """生日号占比: |{x ≤ 31}| / 6。"""
    return sum(1 for x in reds if x <= 31) / 6.0


def _rule_all_same_parity(reds: Sequence[int]) -> float:
    """全奇/全偶 = 1, 否则 0。"""
    odds = sum(1 for x in reds if x % 2 == 1)
    return 1.0 if odds in (0, len(reds)) else 0.0


def _rule_same_tail_pairs(reds: Sequence[int]) -> float:
    """同尾数对计数 / C(6,2)=15: 个位相同的两两组合数。"""
    tails = Counter(x % 10 for x in reds)
    cnt = sum(v * (v - 1) // 2 for v in tails.values())
    return cnt / 15.0


def _rule_lucky_taboo(reds: Sequence[int]) -> float:
    """忌4喜6/8/9: (count(6,8,9) − count(4)) / 6, 截断到 [0,1]。"""
    lucky = sum(1 for x in reds if x in (6, 8, 9))
    taboo = sum(1 for x in reds if x == 4)
    return _clamp01((lucky - taboo) / 6.0)


_RULES: List[tuple] = [
    ("consecutive_pairs", _rule_consecutive_pairs),
    ("arithmetic_progressions", _rule_arithmetic_progressions),
    ("birthday_ratio", _rule_birthday_ratio),
    ("all_same_parity", _rule_all_same_parity),
    ("same_tail_pairs", _rule_same_tail_pairs),
    ("lucky_taboo", _rule_lucky_taboo),
]


def combo_popularity(reds: Sequence[int], weights: Optional[dict] = None) -> float:
    """组合流行度: 返回 [0,1] 归一化得分, 越高=越可能被其他彩民选择(越该 penalize)。

    Args:
        reds: 6 个红球号码(1..33)。
        weights: 可选规则权重覆盖(按规则名部分覆盖默认权重; 权重≤0 的规则跳过)。

    Returns:
        归一化流行度得分, 范围 [0,1], 越高越可能被其他彩民选择。
    """
    w = {**DEFAULT_WEIGHTS, **(weights or {})}
    num = 0.0
    den = 0.0
    for name, fn in _RULES:
        wi = float(w.get(name, 0.0))
        if wi <= 0:
            continue
        num += wi * fn(reds)
        den += wi
    if den <= 0:
        return 0.0
    return _clamp01(num / den)


def popularity_penalty(reds: Sequence[int], lambda_: float = 0.3,
                       weights: Optional[dict] = None) -> float:
    """流行度惩罚系数: 1 - λ·combo_popularity(...), 截断到 [0,1]。

    Args:
        reds: 6 个红球号码(1..33)。
        lambda_: 惩罚强度, 越大热门组合被采中概率越低。
        weights: 规则权重覆盖(透传 combo_popularity)。

    Returns:
        惩罚系数, 范围 [0,1], 越高越易被采样选中。
    """
    pop = combo_popularity(reds, weights)
    return max(0.0, min(1.0, 1.0 - lambda_ * pop))


def _sample_candidate(red_prob: np.ndarray, rng, temperature: float) -> List[int]:
    """温度采样 6 个不重复红球, 约束奇偶/大小比 (逻辑与 select_numbers._sample_red 一致)。

    奇偶比 ∈ {2:4, 3:3, 4:2}、大小比(1-16 小/17-33 大)∈ {2:4, 3:3, 4:2},
    200 次重试, 兜底取概率 Top6。
    """
    for _ in range(200):
        p = np.power(red_prob, 1.0 / temperature)
        p = p / p.sum()
        picks = rng.choice(33, size=6, replace=False, p=p) + 1
        odds = sum(1 for x in picks if x % 2 == 1)
        big = sum(1 for x in picks if x > 16)
        if odds in (2, 3, 4) and big in (2, 3, 4):
            return sorted(picks.tolist())
    # 兜底: 直接取 Top6
    return sorted((np.argsort(red_prob)[-6:] + 1).tolist())


def sample_with_popularity(
    red_prob: np.ndarray,
    rng,
    temperature: float = 0.6,
    lambda_: float = 0.3,
    weights: Optional[dict] = None,
    n_candidates: int = 200,
    popularity_fn: Optional[Callable[[Sequence[int]], float]] = None,
) -> List[int]:
    """冷门组合加权采样: 生成 n_candidates 个候选注, 按流行度惩罚权重重采样 1 注。

    步骤:
      1) 温度采样生成 n_candidates 个候选 6 号注(奇偶/大小比约束)。
      2) 计算各注 popularity_penalty (或传入的自定义 popularity_fn)。
      3) 按 penalty 权重重采样选 1 注(penalty 越高越易被选中)。
      4) 兜底: 候选全 penalty 为 0 时取首注。

    Args:
        red_prob: 33 维红球概率(未归一化亦可)。
        rng: numpy 随机数生成器(default_rng / RandomState)。
        temperature: 温度采样温度(>0)。
        lambda_: 流行度惩罚系数。
        weights: 规则权重覆盖(透传 combo_popularity)。
        n_candidates: 候选注数。
        popularity_fn: 自定义流行度函数(reds -> float), 默认 None 走内置
            popularity_penalty; 供 select_numbers._sample_red 扩展复用。

    Returns:
        选中的 6 个互异红球号码列表(升序无关, 受控随机)。
    """
    n_candidates = max(1, int(n_candidates))
    cands = [_sample_candidate(red_prob, rng, temperature) for _ in range(n_candidates)]
    penal = []
    for c in cands:
        if popularity_fn is not None:
            p = float(popularity_fn(c))
        else:
            p = popularity_penalty(c, lambda_=lambda_, weights=weights)
        penal.append(_clamp01(p))
    total = sum(penal)
    if total <= 0:
        return cands[0]  # 兜底: 候选全 penalty 为 0 时取首注
    probs = np.asarray(penal, dtype=float) / total
    idx = int(rng.choice(n_candidates, p=probs))
    return cands[idx]
