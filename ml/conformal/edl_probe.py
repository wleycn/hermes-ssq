#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Evidential Deep Learning 先验实验 (C2, 研究简报 2026-08-18 [3])。

⚠ 定位 = **先验区分度实验**，非生产包装层。
原因（qwen3.8-max 审核要点，已采纳）：EDL 在 i.i.d. 数据上"证据量可能退化为
常数"，若无可区分度则仅作解释层。本脚本在接入 select_numbers 之前，先用小
Dirichlet 回归探针验证"认知不确定度是否真有区分度"。

做法：
  - 用历史开奖训练一个**证据输出**探针：对每个球种，把近期窗口编码为特征，
    输出 Dirichlet 浓度参数 α（evidence = Σα）。认知不确定度 = K / α₀
    （K=类别数，α₀=Σα）。高证据=低不确定（模型确信），低证据=高不确定。
  - 实验问题：在 i.i.d. 数据上，evidence 是否对"开奖号落在概率头部 vs 尾部"
    有系统性区分？若有 → EDL 可作选号 UQ 分层；若无（预期）→ evidence 退化为
    常数，EDL 仅作解释层可选组件，不进生产。

诚实声明：预期 i.i.d. 下 evidence 无区分度（退化为常数）。本实验的意义是
**用数据验证 qwen 的担忧是否成立**，而非预设结论。

依赖：仅 numpy。
"""
from __future__ import annotations

import numpy as np
from dataclasses import dataclass
from typing import Dict, List, Sequence


@dataclass
class EDLResult:
    ball_type: str
    mean_evidence: float
    std_evidence: float
    evidence_auroc: float              # evidence 对"号码是否头部命中"的 AUROC
    discriminative: bool               # std 小且 AUROC≈0.5 → 无区分度
    verdict: str


def _relu_evidence(logits: np.ndarray) -> np.ndarray:
    """把 logit 过 ReLU 得非负浓度参数 α=relu(logit)+1。"""
    return np.maximum(logits, 0.0) + 1.0


def train_evidence_probe(features: np.ndarray, targets: np.ndarray,
                         n_epochs: int = 50, lr: float = 0.05) -> np.ndarray:
    """极简线性证据探针：W: (n_feat, K) → α = relu(XW)+1。

    用"证据损失"（Sensoy 2018）：L = 分类交叉熵(msoft) + 不确定性正则。
    返回训练后权重 W。
    """
    X = np.asarray(features, dtype=float)
    Y = np.asarray(targets, dtype=int)            # 0..K-1 类别标签
    n, d = X.shape
    K = int(Y.max()) + 1
    W = np.random.default_rng(0).standard_normal((d, K)) * 0.01
    for _ in range(n_epochs):
        logits = X @ W
        alpha = _relu_evidence(logits)
        alpha0 = alpha.sum(axis=1, keepdims=True)
        # 软化 one-hot 期望
        p = alpha / alpha0
        # 交叉熵
        ce = -np.log(p[np.arange(n), Y] + 1e-9).mean()
        # 不确定性正则：鼓励高证据（α0 大）
        reg = ((K / alpha0).mean()) * 0.1
        loss = ce + reg
        # 简化梯度（数值近似更新）
        grad = np.zeros_like(W)
        for i in range(n):
            err = p[i] - np.eye(K)[Y[i]]
            grad += np.outer(X[i], err)
        W -= lr * grad / n
    return W


def extract_features(series: Sequence[float], window: int = 30) -> np.ndarray:
    """把单球序列切成滑动窗口特征（末值、均值、近期频次等）。"""
    x = np.asarray(series, dtype=float).ravel()
    feats = []
    for i in range(window, len(x)):
        w = x[i - window:i]
        feats.append([w.mean(), w.std() if w.std() > 0 else 0.0,
                      (w == int(x[i])).mean(), x[i] / 33.0])
    return np.array(feats)


def run_edl_experiment(red_series: Sequence[int], blue_series: Sequence[int],
                       window: int = 30) -> List[EDLResult]:
    """对红/蓝跑 EDL 区分度先验实验。"""
    out: List[EDLResult] = []
    for name, series, K in (("red", red_series, 33), ("blue", blue_series, 16)):
        s = np.asarray(series, dtype=int)
        feats = extract_features(s, window)
        targets = s[window:]
        if len(targets) < 50:
            out.append(EDLResult(name, 0.0, 0.0, 0.5, False,
                                 "样本不足, 跳过"))
            continue
        W = train_evidence_probe(feats, targets, n_epochs=30)
        logits = feats @ W
        alpha = _relu_evidence(logits)
        evidence = alpha.sum(axis=1)
        # 区分度：evidence 能否预测"开奖号是否落在模型 top-K/3 概率头部"
        probs = alpha / alpha.sum(axis=1, keepdims=True)
        head = np.array([1 if probs[i, targets[i]] >= (1.0 / K) * 1.5 else 0
                         for i in range(len(targets))])
        # AUROC 近似（按 evidence 排序，简单符次序相关）
        auroc = _approx_auroc(evidence, head)
        mean_e, std_e = float(evidence.mean()), float(evidence.std())
        discriminative = (std_e > 1e-6) and (abs(auroc - 0.5) > 0.05)
        out.append(EDLResult(
            name, round(mean_e, 3), round(std_e, 3), round(auroc, 3),
            discriminative,
            ("evidence 有区分度, 可作 UQ 分层" if discriminative
             else "evidence 无区分度(退化为常数), 仅作解释层")))
    return out


def _approx_auroc(scores: np.ndarray, labels: np.ndarray) -> float:
    """AUROC 近似（Mann-Whitney U）。"""
    scores = np.asarray(scores, dtype=float)
    labels = np.asarray(labels, dtype=int)
    pos = scores[labels == 1]
    neg = scores[labels == 0]
    if pos.size == 0 or neg.size == 0:
        return 0.5
    cnt = sum(1 for p in pos for n in neg if p > n)
    cnt += 0.5 * sum(1 for p in pos for n in neg if p == n)
    return cnt / (pos.size * neg.size)


def summarize_edl(results: List[EDLResult]) -> dict:
    n_disc = sum(1 for r in results if r.discriminative)
    return {
        "n_ball_types": len(results),
        "n_discriminative": n_disc,
        "overall": "EDL_USABLE" if n_disc > 0 else "EDL_EXPLAIN_ONLY",
        "recommendation": (
            "EDL 可作 select_numbers 上游 UQ 包装层" if n_disc > 0
            else "EDL 证据量在 i.i.d. 下无区分度 → 仅作解释层可选组件，不进生产"),
    }
