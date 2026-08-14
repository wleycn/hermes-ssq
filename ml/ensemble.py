#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""轻量集成融合: 等权均值 vs EBMA(Bayesian Model Averaging 近似)。

为什么有这个模块(Rocky 指示 2026-08-13, 源自简报 EBMA / BayesBlend):
  select_numbers.py 原先对多模型概率做**等权均值**。EBMA 给出"有后验权重的严谨
  融合"——权重由各模型在验证期(历史开奖)上的表现校准, 而非拍脑袋。

重要诚实声明: 双色球为独立均匀随机过程, 更严谨的集成 **不等于** 更高命中率
(数学上不可能)。本模块的价值是"融合过程可复现、可报告不确定度", 回应 Rocky
对统计严谨性的要求, 而非提升预测精度。

实现选择(零依赖):
  不装 bayesblend(它拉 cmdstanpy 100MB+ + 固定 matplotlib 3.7.2, 性价比低)。
  自实现轻量 EBMA:
    - 每个模型 m 在验证期 D 上的 ELPD 代理 = Σ_{d∈D} log P_m(draw_d)
      (即该模型概率分布下, 实际开奖号码的对数概率之和)
    - 权重 w_m = softmax(ELPD_m / tau), tau 为温度(默认 1.0, 等价于伪 BMA)
      tau→∞ 退化为等权; 越小权重越集中于高 ELPD 模型
    - 集成概率 = Σ_m w_m * P_m

用法:
  from ml.ensemble import integrate
  red_mean = integrate(red_models_dict, history, ball="red", method="ebma")
"""
from __future__ import annotations

import csv
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
CSV_PATH = ROOT / "ml" / "data" / "1.csv"


def _load_history_redblue(path: Path) -> List[Tuple[np.ndarray, int]]:
    """读取 1.csv, 返回 [(red_probs_anchor_ignored, blue), ...] 的 (reds_set, blue)。

    这里只需要"实际开奖"来算各模型历史 log-likelihood, 不依赖模型自身的预测。
    """
    rows = []
    with open(path, encoding="utf-8-sig", newline="") as f:
        for row in csv.DictReader(f):
            reds = frozenset(int(row[f"Red{i}"]) for i in range(1, 7))
            blue = int(row["Blue1"])
            rows.append((reds, blue))
    return rows


def _model_loglik(prob: np.ndarray, history, ball: str) -> float:
    """模型在某侧(红/蓝)历史开奖上的对数似然和(ELPD 代理)。

    Args:
        prob: 该模型在 ball 侧的 33 维(红)或 16 维(蓝)概率向量。
        history: [(reds_set, blue), ...]
        ball: "red" 或 "blue"。

    Returns:
        Σ log P(draw | model)。概率过小时加 epsilon 防 log(0)。
    """
    eps = 1e-12
    p = np.asarray(prob, dtype=float)
    p = np.clip(p, eps, 1.0)
    ll = 0.0
    for reds_set, blue in history:
        if ball == "red":
            # 实际开奖 6 个红球各自概率之和的对数(独立假设)
            ll += float(np.sum(np.log(p[np.array(list(reds_set)) - 1])))
        else:
            ll += float(np.log(p[blue - 1]))
    return ll


def integrate(
    models_prob: Dict[str, np.ndarray],
    history_path: Path = CSV_PATH,
    ball: str = "red",
    method: str = "mean",
    tau: float = 1.0,
) -> Tuple[np.ndarray, Dict[str, float]]:
    """多模型概率融合。

    Args:
        models_prob: {model_name: prob_vector(33 或 16 维)}。
        history_path: 历史开奖 CSV(用于 EBMA 权重校准)。
        ball: "red" 或 "blue"。
        method: "mean"(等权) 或 "ebma"(softmax 加权)。
        tau: EBMA 温度; tau→∞ 退化为等权, 越小越集中于高 ELPD 模型。

    Returns:
        (integrated_prob, weights) 元组。weights 为各模型权重(dict)。
    """
    names = list(models_prob.keys())
    if not names:
        raise ValueError("models_prob 为空")
    dim = len(models_prob[names[0]])
    mats = np.stack([np.asarray(models_prob[n], dtype=float) for n in names])  # (M, dim)

    if method == "mean":
        weights = {n: 1.0 / len(names) for n in names}
        integrated = mats.mean(axis=0)
        return integrated, weights

    if method == "ebma":
        history = _load_history_redblue(history_path)
        ell = np.array([_model_loglik(models_prob[n], history, ball) for n in names])
        # softmax(ELPD / tau): tau 控制集中度
        scaled = ell / max(tau, 1e-9)
        scaled -= scaled.max()  # 数值稳定
        w = np.exp(scaled)
        w = w / w.sum()
        weights = {n: float(ww) for n, ww in zip(names, w)}
        integrated = (w[:, None] * mats).sum(axis=0)
        return integrated, weights

    raise ValueError(f"未知 method={method}, 仅支持 mean/ebma")


def integrate_redblue(
    red_models: Dict[str, np.ndarray],
    blue_models: Dict[str, np.ndarray],
    history_path: Path = CSV_PATH,
    method: str = "mean",
    tau: float = 1.0,
) -> Tuple[np.ndarray, np.ndarray, Dict[str, float], Dict[str, float]]:
    """ Convenience: 红蓝两侧分别融合, 返回 (red_mean, blue_mean, red_w, blue_w)。"""
    red_mean, red_w = integrate(red_models, history_path, "red", method, tau)
    blue_mean, blue_w = integrate(blue_models, history_path, "blue", method, tau)
    return red_mean, blue_mean, red_w, blue_w
