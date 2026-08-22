#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""James-Stein 收缩后处理 (研究简报 2026-08-22 [1])。

定位：概率后处理层。对「6 模型均值集成后的概率向量」向均匀先验做收缩，
降低有限样本 MSE——本质是**对抗模型在随机数据上学到的过拟合噪声**的正则化
（qwen 2026-08-22 审核明确：Stein 的价值是正则化而非预测信号增强）。

原理（James & Stein 1961; Efron《Large-Scale Inference》2010）：
  当待估参数维度 d≥3 时，向公共先验 μ₀ 收缩的估计器
      θ̂_JS = (1 − c)·θ̂ + c·μ₀,   c = (d − 2)·σ² / ‖θ̂ − μ₀‖²
  在 MSE 意义下**严格优于**直接用 θ̂（Stein 悖论）。对 SSQ：
  - 红球: θ̂ ∈ R³³, μ₀ = 1/33 均匀向量；
  - 蓝球: θ̂ ∈ R¹⁶, μ₀ = 1/16 均匀向量。

与 EBMA 正交：EBMA 是「多模型权重」平均，Stein 是「单概率向量」本身修正。

诚实预告：i.i.d. 随机下模型输出本应≈均匀，收缩只是把噪声压回均匀，
**不提升命中率**。止损线：walk-forward 无显著区分 → rollback。
依赖：仅 numpy。
"""
from __future__ import annotations

from typing import Sequence

import numpy as np


def james_stein_shrink(
    probs: Sequence[float] | np.ndarray,
    prior: Sequence[float] | np.ndarray | None = None,
    sigma2: float = 1.0,
    alpha: float = 1.0,
) -> np.ndarray:
    """James-Stein 收缩：向先验(默认均匀)收缩概率向量。

    Args:
        probs: 概率向量 θ̂（如 33 维红球概率，需已归一化或未归一化均可，
            收缩后自动重归一化）。
        prior: 公共先验 μ₀；None 时取均匀（1/d 每分量）。
        sigma2: 每分量的噪声方差估计 σ²；默认 1.0（保守，收缩温和）。
        alpha: 收缩强度系数 ∈[0,1]；1=标准 JS，0=不收缩（纯原样）。
    Returns:
        收缩并重归一化后的概率向量（形状同 probs）。
    """
    p = np.asarray(probs, dtype=float).ravel()
    d = p.size
    if d < 3:
        raise ValueError(f"James-Stein 要求 d≥3, 实际 d={d}")
    if alpha <= 0:
        s = p.copy()
    else:
        if prior is None:
            mu = np.full(d, 1.0 / d)
        else:
            mu = np.asarray(prior, dtype=float).ravel()
            if mu.size != d:
                raise ValueError(f"prior 维度 {mu.size} != probs 维度 {d}")
        diff = p - mu
        norm2 = float(np.sum(diff ** 2))
        if norm2 <= 0:
            c = 0.0  # 已等于先验, 无需收缩
        else:
            c = alpha * (d - 2) * sigma2 / norm2
        c = min(c, 1.0)  # 收缩因子封顶 1(不越过先验)
        s = (1.0 - c) * p + c * mu
    total = s.sum()
    if total <= 0:
        return np.full(d, 1.0 / d)
    return s / total


def shrink_red_blue(red_prob: Sequence[float] | np.ndarray,
                    blue_prob: Sequence[float] | np.ndarray,
                    sigma2: float = 1.0, alpha: float = 1.0) -> tuple[np.ndarray, np.ndarray]:
    """对红(33)/蓝(16)集成概率分别做 James-Stein 收缩。

    Returns: (red_js, blue_js)，均已归一化。
    """
    red_js = james_stein_shrink(red_prob, sigma2=sigma2, alpha=alpha)
    blue_js = james_stein_shrink(blue_prob, sigma2=sigma2, alpha=alpha)
    return red_js, blue_js


if __name__ == "__main__":  # pragma: no cover
    # 演示: 模型输出轻微偏离均匀 → 收缩后更接近均匀
    rng = np.random.default_rng(1)
    p = rng.dirichlet(np.full(33, 0.9))
    js, _ = shrink_red_blue(p, np.full(16, 1 / 16))
    print("原最大概率: %.4f -> 收缩后: %.4f" % (p.max(), js.max()))
