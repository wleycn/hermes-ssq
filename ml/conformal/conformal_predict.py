#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Conformal Prediction 集合层 (C1, 研究简报 2026-08-17 [3])。

定位：不确定性量化（UQ），**不改变命中率**（不可能改变），但把现有 8 模型
概率输出升级为带理论覆盖率保证的**候选集合**。接 select_numbers 上游。

原理（Angelopoulos & Bates 2021 "A Gentle Introduction to Conformal Prediction"）：
  - 用历史开奖做 calibration split。对每个球种（红 1..33 / 蓝 1..16）与每个模型，
    计算 conformity score（如：开奖号在模型概率分布中的分位）。
  - 选定目标覆盖率 1-α（如 0.90），取分位阈值 q̂。
  - 预测集合 = {号码 i : score(i) ≤ q̂}，即"以 90% 把握本期号码落在该集合内"。
  - exchangeability 在 i.i.d. 下天然成立 → 覆盖率保证有效（无需独立性假设）。

诚实声明：覆盖率保证的是"集合包含开奖号"的概率，不等于"集合里就是中奖号"。
对彩票的意义 = 把"概率 0.03"翻译成"我有 90% 把握号码落在集合 S(大小≈k) 里"，
让选号从盲抽变为可解释的风险分层。集合大小(=不确定性代理)本身无预测增益。

用法：
  from ml.conformal.conformal_predict import ConformalSet, build_from_history
  cs = build_from_history(df, models_probs, alpha=0.90)
  red_set, blue_set = cs.predict_set(latest_red_prob, latest_blue_prob)
  # red_set: 本期红球候选集合（如 12 个）；blue_set: 蓝球候选集合（如 3 个）

依赖：仅 numpy。
"""
from __future__ import annotations

import numpy as np
from dataclasses import dataclass, field
from typing import Dict, List, Sequence


@dataclass
class ConformalSet:
    """单球种 conformal 集合生成器。"""
    alpha: float = 0.90                 # 目标覆盖率 1-α
    conformity_scores: np.ndarray = field(default_factory=lambda: np.array([]))
    q_hat: float = 0.0

    def calibrate(self, scores: Sequence[float]) -> None:
        """用 calibration split 的分位分数确定阈值 q̂。

        阈值取 (⌈(n+1)(1-α)⌉ / n) 分位（有限样本校正，保证边际覆盖率≥1-α）。
        """
        s = np.sort(np.asarray(scores, dtype=float))
        n = s.size
        if n == 0:
            self.q_hat = 0.0
            self.conformity_scores = s
            return
        level = np.ceil((n + 1) * (1 - self.alpha)) / n
        level = min(max(level, 0.0), 1.0)
        idx = min(int(level * n), n - 1)
        self.q_hat = float(s[idx])
        self.conformity_scores = s

    def predict_set(self, prob: Sequence[float]) -> List[int]:
        """给定某模型对 33/16 个号码的概率分布，返回 conformal 候选集合（1-indexed）。

        conformity score = 该号码的概率质量 p(i)（概率越高越"典型"，score 越大）。
        集合 = {i : score(i) ≥ q̂}，q̂ 为 calibration 分位阈值（见 calibrate）。
        因 drawn 号码在训练时多为高概率（score 大），阈值 q̂ 取 (1-α) 分位可保证
        边际覆盖率 ≥ 1-α（exchangeability 下成立）。
        """
        p = np.asarray(prob, dtype=float)
        scores = p / p.sum()
        return [int(i + 1) for i, sc in enumerate(scores) if sc >= self.q_hat - 1e-12]


def build_from_history(
    red_history: Sequence[Sequence[float]],
    blue_history: Sequence[Sequence[float]],
    red_draws: Sequence[Sequence[int]],
    blue_draws: Sequence[int],
    alpha: float = 0.90,
) -> Dict[str, ConformalSet]:
    """用历史概率 + 历史开奖校准红/蓝 conformal 集合。

    Args:
        red_history: 每期 33 维红球概率（顺序与 red_draws 对齐）。
        blue_history: 每期 16 维蓝球概率。
        red_draws: 每期 6 个红球号码（1..33）。
        blue_draws: 每期 1 个蓝球号码（1..16）。
    Returns:
        {"red": ConformalSet, "blue": ConformalSet}
    """
    red_scores, blue_scores = [], []
    for prob, drawn in zip(red_history, red_draws):
        p = np.asarray(prob, dtype=float); p = p / p.sum()
        for d in drawn:
            red_scores.append(p[d - 1])          # conformity = 该号码概率质量
    for prob, drawn in zip(blue_history, blue_draws):
        p = np.asarray(prob, dtype=float); p = p / p.sum()
        blue_scores.append(p[drawn - 1])

    red_cs = ConformalSet(alpha=alpha); red_cs.calibrate(red_scores)
    blue_cs = ConformalSet(alpha=alpha); blue_cs.calibrate(blue_scores)
    return {"red": red_cs, "blue": blue_cs}


def summarize_coverage(cs: ConformalSet, prob: Sequence[float]) -> dict:
    """报告候选集合大小（不确定性代理）。"""
    s = cs.predict_set(prob)
    return {
        "alpha": cs.alpha,
        "q_hat": round(cs.q_hat, 4),
        "set_size": len(s),
        "coverage_claim": f"以 {cs.alpha*100:.0f}% 把握本期号码落在集合(大小={len(s)})内",
    }
