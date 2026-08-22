#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""RQA 递归量化分析探针 (研究简报 2026-08-19 [2], 2026-08-22 补落地)。

定位：随机性判据（递归域/相位空间家族）。把序列做时间延迟嵌入重建相位
轨迹, 计算递归图 R(i,j)=Θ(ε−‖x(i)−x(j)‖), 量化:
  - RR  递归率（递归点占比）
  - DET 确定性（构成对角线的递归点占比, 随机序列低, 确定性动力学高）
  - Lmax 最长对角线长度（周期/确定性结构的标志）
预期：随机序列预期: RR 低、DET 低、Lmax 短; 显著偏离 → 非随机动力学。

适用性边界 (2026-08-22 实测): 同可见图——适用于 i.i.d. 序列(蓝球/和值/
单号);"从 33 取 6 不放回"的红球全量展平会误报 NONRANDOM(组合约束伪影)。
纯 numpy 实现(ε 用距离分布分位数自适应);O(n²) 递归矩阵, 长序列截断。
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import List, Sequence

import numpy as np

from ml.probes.surrogate_probe import make_surrogates, surrogate_zscore


@dataclass
class RqaResult:
    name: str
    metric: str            # det | lmax
    value: float
    verdict: str           # RANDOM | NONRANDOM | INVALID
    detail: str = ""


def _embed(x: np.ndarray, dim: int = 3, tau: int = 1) -> np.ndarray:
    """时间延迟嵌入: 返回 (n-(dim-1)*tau, dim) 轨迹矩阵。"""
    n = x.size
    if n < dim * tau + 1:
        raise ValueError(f"序列过短 (n={n}) 无法嵌入 dim={dim}")
    idx = np.arange(n - (dim - 1) * tau)
    return np.stack([x[idx + i * tau] for i in range(dim)], axis=1)


def _recurrence_matrix(tr: np.ndarray, eps: float | None = None) -> np.ndarray:
    """递归图 R(i,j)=Θ(ε−‖x(i)−x(j)‖)。ε 默认距离分布的 10% 分位数。"""
    from scipy.spatial.distance import pdist, squareform
    d = squareform(pdist(tr, metric="euclidean"))
    if eps is None:
        eps = np.percentile(d[d > 0], 10.0)
    return (d <= eps).astype(np.float64)


def _det_lmax(rm: np.ndarray) -> tuple[float, float]:
    """从递归图算 DET(对角线递归占比) 与 Lmax(最长对角线, 含主对角)。"""
    n = rm.shape[0]
    # 主对角线上递归点不算(自递归)
    diag_mask = np.eye(n, dtype=bool)
    total_rec = rm.sum() - n  # 去掉主对角线
    # 统计所有对角线(偏移 d=1..n-1)的连续递归段长
    max_run = 0
    diag_run = 0.0
    for d in range(1, n):
        seg = np.diag(rm, k=d)
        # 连续 1 段
        runs = np.diff(np.where(np.concatenate(([0], seg, [0])) == 0)[0]) - 1
        if runs.size:
            max_run = max(max_run, int(runs.max()))
            diag_run += float(np.sum(runs[runs >= 2]))
    if total_rec <= 0:
        return 0.0, 0.0
    det = diag_run / total_rec
    return det, float(max_run)


def run_rqa_probe(x: Sequence[float] | np.ndarray, dim: int = 3, tau: int = 1,
                  n_surrogates: int = 30, seed: int = 42) -> List[RqaResult]:
    """对序列跑 RQA, DET/Lmax 相对 RS surrogate 判显著。"""
    arr = np.asarray(x, dtype=float).ravel()
    if arr.size < dim * tau + 20:
        return [RqaResult("rqa_det", "det", 0.0, "INVALID", "序列过短")]
    max_points = 600  # RQA 递归矩阵 O(n²), 长序列截断(检验器, 600 点足够)
    if arr.size > max_points:
        arr = arr[-max_points:]
    tr = _embed(arr, dim=dim, tau=tau)
    rm = _recurrence_matrix(tr)
    det, lmax = _det_lmax(rm)

    surros = make_surrogates(arr, kind="rs", n=n_surrogates, seed=seed)

    def _stat_det(s: np.ndarray) -> float:
        try:
            return _det_lmax(_recurrence_matrix(_embed(s, dim, tau)))[0]
        except Exception:
            return float("nan")

    def _stat_lmax(s: np.ndarray) -> float:
        try:
            return _det_lmax(_recurrence_matrix(_embed(s, dim, tau)))[1]
        except Exception:
            return float("nan")

    z_det = surrogate_zscore(arr, _stat_det, surros)
    z_lmax = surrogate_zscore(arr, _stat_lmax, surros)

    return [
        RqaResult("rqa_det", "det", round(float(det), 4),
                  "RANDOM" if abs(z_det) < 2 else "NONRANDOM",
                  f"z={z_det:.2f}; 确定性, 随机预期低"),
        RqaResult("rqa_lmax", "lmax", round(float(lmax), 2),
                  "RANDOM" if abs(z_lmax) < 2 else "NONRANDOM",
                  f"z={z_lmax:.2f}; 最长对角线, 随机预期短"),
    ]


if __name__ == "__main__":  # pragma: no cover
    rng = np.random.default_rng(7)
    demo = rng.standard_normal(300)
    for r in run_rqa_probe(demo, n_surrogates=15):
        print(f"{r.name:10s} {r.metric:8s} value={r.value:8.4f} {r.verdict:8s} {r.detail}")
