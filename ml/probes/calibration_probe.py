#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""概率校准诊断探针 (研究简报 2026-08-22 [2])。

定位：Brier / reliability 监控（诊断层，非生产选号路径）。
对「模型输出概率 vs 真实开奖」评估概率诚实度：
  - Brier score（概率与 one-hot 的均方误差，越低越诚实）；
  - Reliability 曲线/ECE（Expected Calibration Error）：把概率分桶，
    比较「平均预测概率」与「实际频率」，差距=校准误差；
  - Isotonic 校准前后对比：预期 i.i.d. 均匀输出下 isotonic≈恒等映射
    （qwen 2026-08-22 审核与 08-22 简报共同盲区：校准在均匀输出上
    收益≈0），故本模块只做监控，不宣称命中率提升。

用法（探针级，随月度重训/研究 cron 运行）：
  from ml.probes.calibration_probe import evaluate_calibration
  res = evaluate_calibration(probs_history, draws_history, n_bins=10)
依赖：仅 numpy。
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import List, Sequence

import numpy as np


@dataclass
class CalibrationResult:
    metric: str              # brier | ece | isotonic_delta
    value: float
    detail: str = ""


def brier_score(probs: Sequence[float] | np.ndarray, outcome: int) -> float:
    """单期 Brier: 概率向量 vs one-hot(开奖号位置=1)。

    Args:
        probs: 概率向量(如 33 红 或 16 蓝)。
        outcome: 1-indexed 开奖号。
    """
    p = np.asarray(probs, dtype=float).ravel()
    y = np.zeros_like(p)
    y[outcome - 1] = 1.0
    return float(np.mean((p - y) ** 2))


def ece(probs: Sequence[float] | np.ndarray, outcomes: Sequence[int],
        n_bins: int = 10) -> float:
    """Expected Calibration Error：展平(样本×类别)二元口径分桶比较。

    正确口径（2026-08-22 实现修正）：把所有 (样本 i, 类别 j) 对的预测概率
    p_ij 与是否开出 y_ij∈{0,1} 一起分桶，比较「平均预测概率」vs「实际
    开出频率」。此前误用「开奖号位置概率 vs 频率=1」口径导致 ECE 恒≈0.97。

    Args:
        probs: 二维数组 (n_samples, n_classes)，每行为一个概率向量。
        outcomes: 1-indexed 开奖号列表，长度 n_samples。
    """
    P = np.asarray(probs, dtype=float)
    if P.ndim != 2:
        raise ValueError(f"probs 应为 2 维 (n, d), 实际 {P.ndim} 维")
    y = np.asarray(outcomes, dtype=int)
    if P.shape[0] != y.size:
        raise ValueError("probs 与 outcomes 长度不一致")
    n, d = P.shape
    # 展平: 每个 (i, j) 一对
    conf = P.ravel()
    ymat = np.zeros((n, d), dtype=float)
    ymat[np.arange(n), y - 1] = 1.0
    acc = ymat.ravel()
    bins = np.linspace(0, 1, n_bins + 1)
    ece_val = 0.0
    for i in range(n_bins):
        lo, hi = bins[i], bins[i + 1]
        m = (conf >= lo) & (conf < hi)
        if i == n_bins - 1:  # 最后一桶含右端点
            m = (conf >= lo) & (conf <= hi)
        if m.sum() == 0:
            continue
        ece_val += (m.sum() / conf.size) * abs(conf[m].mean() - acc[m].mean())
    return float(ece_val)


def isotonic_delta(probs: Sequence[float] | np.ndarray,
                   outcomes: Sequence[int]) -> float:
    """Isotonic 回归校准前后 Brier 变化(预期≈0, 说明校准无操作空间)。

    展平(样本×类别)二元口径：所有 p_ij vs y_ij 作为校准样本。
    用留一法避免过拟合: 每个样本用其余样本拟合的 isotonic 映射。
    """
    P = np.asarray(probs, dtype=float)
    y = np.asarray(outcomes, dtype=int)
    n, d = P.shape
    if n < 5:
        raise ValueError("样本不足, isotonic 至少需 5 个")
    ymat = np.zeros((n, d), dtype=float)
    ymat[np.arange(n), y - 1] = 1.0
    conf = P.ravel()
    acc = ymat.ravel()
    if conf.size < 8:
        raise ValueError("展平后样本不足")
    # 留一 O(n²), 全量 16500 点过慢 → 抽样子集(300)估计, 足够稳定
    rng = np.random.default_rng(0)
    idx = rng.choice(conf.size, size=min(300, conf.size), replace=False)
    conf_s = conf[idx]
    acc_s = acc[idx]
    raw_brier = np.mean((conf_s - acc_s) ** 2)
    cal_loo = np.empty(conf_s.size)
    for i in range(conf_s.size):
        mask = np.ones(conf_s.size, dtype=bool)
        mask[i] = False
        cal_loo[i] = _pav(conf_s[mask], acc_s[mask])(conf_s[i])
    cal_brier = np.mean((cal_loo - acc_s) ** 2)
    return float(cal_brier - raw_brier)


def _pav(values: np.ndarray, targets: np.ndarray | None = None):
    """Pool Adjacent Violators: 返回单调非降插值函数(isotonic 回归)。

    栈式 PAV (标准实现, O(n log n)): 维护单调块栈, 每压入一个新块,
    若与前一块均值倒挂则合并, 直到全栈单调。

    Args:
        values: 预测值(函数内按升序重排)。
        targets: 对应监督值(默认 = values 本身, 即对 values 做保序回归)。
    Returns:
        单调插值函数 f(q)。
    """
    v = np.asarray(values, dtype=float).ravel()
    t = np.asarray(values, dtype=float).ravel() if targets is None \
        else np.asarray(targets, dtype=float).ravel()
    if v.size != t.size or v.size == 0:
        raise ValueError("values 与 targets 长度须一致且非空")
    order = np.argsort(v, kind="stable")
    v = v[order]
    t = t[order]

    # 栈: 每块存 (v_sum, v_n, t_sum, t_n, v_min, v_max)
    stack: list[tuple[float, int, float, int, float, float]] = []
    for vi, ti in zip(v, t):
        block = (vi, 1, ti, 1, vi, vi)
        while stack:
            pv_s, pv_n, pt_s, pt_n, _, _ = stack[-1]
            if pt_s / pt_n <= block[2] / block[3]:
                break
            # 与栈顶合并(均值倒挂)
            bv_s, bv_n, bt_s, bt_n, bmin, bmax = block
            block = (pv_s + bv_s, pv_n + bv_n, pt_s + bt_s, pt_n + bt_n,
                     min(pmin := stack[-1][4], bmin), max(stack[-1][5], bmax))
            stack.pop()
        stack.append(block)

    xs = [b[4] for b in stack] + [stack[-1][5]]
    ys = [b[2] / b[3] for b in stack] + [stack[-1][2] / stack[-1][3]]

    def _interp(q: float) -> float:
        return float(np.interp(q, xs, ys))

    return _interp


def evaluate_calibration(probs_history: Sequence[Sequence[float] | np.ndarray],
                         outcomes: Sequence[int], n_bins: int = 10) -> List[CalibrationResult]:
    """对历史概率批次评估 Brier / ECE / isotonic 前后 Brier 差。

    Args:
        probs_history: 每期的概率向量列表(同球种)。
        outcomes: 对应期的开奖号(1-indexed)。
    Returns:
        三个指标的结果列表。
    """
    P = np.asarray([np.asarray(p, dtype=float).ravel() for p in probs_history])
    y = np.asarray(outcomes, dtype=int)
    n = P.shape[0]
    if n != y.size:
        raise ValueError("probs_history 与 outcomes 长度不一致")
    if n < 5:
        return [CalibrationResult("brier", 0.0, "样本不足"),
                CalibrationResult("ece", 0.0, "样本不足"),
                CalibrationResult("isotonic_delta", 0.0, "样本不足")]
    brier = np.mean([brier_score(P[i], y[i]) for i in range(n)])
    ece_val = ece(P, y, n_bins=n_bins)
    try:
        iso = isotonic_delta(P, y)
    except ValueError:
        iso = 0.0
    return [
        CalibrationResult("brier", round(float(brier), 6)),
        CalibrationResult("ece", round(ece_val, 6),
                          f"n_bins={n_bins}, 越小越接近诚实"),
        CalibrationResult("isotonic_delta", round(iso, 6),
                          "≈0 说明校准无操作空间(i.i.d. 均匀输出预期)"),
    ]


if __name__ == "__main__":  # pragma: no cover
    rng = np.random.default_rng(1)
    probs = np.stack([rng.dirichlet(np.full(33, 1.0)) for _ in range(500)])
    outs = rng.integers(1, 34, size=500)
    for r in evaluate_calibration(probs, outs):
        print(f"{r.metric:16s} = {r.value:.6f}  {r.detail}")
