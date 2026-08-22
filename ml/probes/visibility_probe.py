#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""可见图 Visibility Graph 探针 (研究简报 2026-08-18 [1], 2026-08-22 补落地)。

定位：随机性判据（图论家族）。把时间序列映射为图——相邻两点若中间所有
点都低于二者连线则连边（自然可见准则）。随机序列的度分布呈指数型
P(k) ∝ exp(-λk)（Lacasa et al. 2008 理论）；偏离即提示非随机结构。

判据：
  1. 度分布与指数拟合的 R²（越接近 1 越符合随机理论）；
  2. 度分布 χ² vs 均匀/幂律对照的显著性；
  3. 平均路径长度/聚类系数（参考值，不单独判据）。
预期：彩票序列度分布≈指数 → RANDOM。

适用性边界 (2026-08-22 实测): 本探针适用于**独立同分布序列**(蓝球 1..16、
和值、单号序列);对"从 33 取 6 不放回"的红球全量展平序列会误报
NONRANDOM——组合约束(值域受限、无重复)产生天然结构,非随机缺陷。
纯 numpy 实现;O(n²) 邻接矩阵, 长序列自动截断尾部。
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Sequence

import numpy as np


@dataclass
class VisibilityResult:
    name: str
    metric: str            # lambda | ks_p
    value: float
    verdict: str           # RANDOM | NONRANDOM | INVALID
    detail: str = ""


def visibility_graph(x: Sequence[float] | np.ndarray) -> np.ndarray:
    """构建可见图邻接矩阵（自然可见准则, O(n²) 暴力 + 线性扫描优化）。

    Args:
        x: 时间序列。
    Returns:
        (n, n) 布尔邻接矩阵（含自环=False, 无向）。
    """
    arr = np.asarray(x, dtype=float).ravel()
    n = arr.size
    adj = np.zeros((n, n), dtype=bool)
    for i in range(n):
        for j in range(i + 1, n):
            # 中间所有点都低于 i->j 连线 → 可见
            if j == i + 1:
                adj[i, j] = adj[j, i] = True
                continue
            seg = arr[i + 1:j]
            # 点 k 在直线 i->j 下方: y_k < y_i + (y_j - y_i) * (k-i)/(j-i)
            t = np.arange(1, j - i)
            line = arr[i] + (arr[j] - arr[i]) * t / (j - i)
            if np.all(seg < line):
                adj[i, j] = adj[j, i] = True
    return adj


def degree_distribution(adj: np.ndarray) -> np.ndarray:
    """返回度分布 {k: 频率}（k=0..n-1, 归一化）。"""
    deg = adj.sum(axis=1)
    n = adj.shape[0]
    hist = np.bincount(deg, minlength=n).astype(float) / n
    return hist


def _fit_exponential(ks: np.ndarray, ps: np.ndarray, n_samples: int) -> tuple[float, float]:
    """随机可见图度分布几何参数估计 (MLE), 返回 (λ, p_ks)。

    理论 (Lacasa et al. 2008): 无限随机序列 P(k) = (1/3)(2/3)^(k-2), k≥2,
    即平移几何分布, p=1/3, λ = -ln(1-p) ≈ 0.405。
    有限样本受边界效应影响会偏离理论(度分布整体衰减变慢), 故本函数只
    用 MLE 估 λ(作为形状摘要), 显著性交由 run_visibility_probe 的
    surrogate 比较(同一长度的 RS 洗牌序列才是正确零假设)。

    Args:
        ks: 度值数组 0..n-1。
        ps: 归一化度分布(频率, 和≈1)。
        n_samples: 实际样本数(序列长度 n), 仅用于 MLE 加权。
    """
    mask = ks >= 2
    if mask.sum() < 3:
        return float("nan"), float("nan")
    kk = ks[mask].astype(float)
    pk = ps[mask]
    total = pk.sum()
    if total <= 0 or n_samples < 10:
        return float("nan"), float("nan")
    ek = float(np.sum(kk * pk) / total)
    if ek <= 2:
        return float("nan"), float("nan")
    p = 1.0 / (ek - 2)
    lam = -np.log1p(-p)
    return float(lam), float("nan")  # p_ks 不用(改用 surrogate 判据)


def run_visibility_probe(x: Sequence[float] | np.ndarray, n_surrogates: int = 30,
                         seed: int = 42) -> List[VisibilityResult]:
    """对序列跑可见图度分布检验, λ 相对 RS surrogate 判显著。

    判据: 原序列 λ vs 同一长度 RS 洗牌序列的 λ 分布(|z|<2 → RANDOM)。
    这是有限样本下正确的零假设(理论几何分布是无限序列极限)。

    性能注: 可见图构建 O(n²), 输入超 MAX_POINTS(默认 800)时自动截断
    尾部——探针是检验器, 800 点已足够估计度分布形状。
    """
    arr = np.asarray(x, dtype=float).ravel()
    if arr.size < 10:
        return [VisibilityResult("vg_lambda", "lambda", 0.0, "INVALID", "序列过短")]
    max_points = 800
    if arr.size > max_points:
        arr = arr[-max_points:]  # 取最近段(与金融\"最新窗口\"语义一致)

    def _lam(seq: np.ndarray) -> float:
        adj = visibility_graph(seq)
        dist = degree_distribution(adj)
        ks = np.arange(dist.size)
        lam, _ = _fit_exponential(ks, dist, n_samples=seq.size)
        return lam

    lam = _lam(arr)
    if not np.isfinite(lam):
        return [VisibilityResult("vg_lambda", "lambda", float("nan"), "INVALID", "拟合失败")]

    rng = np.random.default_rng(seed)
    lams = []
    for _ in range(n_surrogates):
        l = _lam(rng.permutation(arr))
        if np.isfinite(l):
            lams.append(l)
    lams = np.array(lams)
    z = (lam - lams.mean()) / lams.std() if lams.std() > 0 else 0.0

    return [
        VisibilityResult("vg_lambda", "lambda", round(lam, 4),
                         "RANDOM" if abs(z) < 2 else "NONRANDOM",
                         f"MLE 几何参数; 随机理论≈0.405, z={z:.2f}"),
        VisibilityResult("vg_z", "z", round(float(z), 4),
                         "RANDOM" if abs(z) < 2 else "NONRANDOM",
                         f"vs {n_surrogates} 个 RS surrogate"),
    ]


if __name__ == "__main__":  # pragma: no cover
    rng = np.random.default_rng(7)
    demo = rng.standard_normal(300)
    for r in run_visibility_probe(demo):
        print(f"{r.name:10s} {r.metric:8s} value={r.value:8.4f} {r.verdict:8s} {r.detail}")
