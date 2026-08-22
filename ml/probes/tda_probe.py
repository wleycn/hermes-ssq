#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""TDA 持久同调探针 (研究简报 2026-08-20 [2], 2026-08-22 补落地)。

定位：随机性判据（代数拓扑家族, 14 家族收官项）。把时间序列做时间延迟
嵌入(Takens)得到高维点云 → 建 Vietoris-Rips 复形 → 计算 persistence
diagram(H0 连通分量 / H1 环 / H2 空洞的"生-灭"尺度)。纯随机噪声的
拓扑特征寿命短且服从已知分布; 有结构序列会出现长命特征。

判据：
  1. H1(环)最大寿命 vs RS surrogate 分布(|z|<2 → RANDOM);
  2. H1 显著长命特征数(寿命超过 95% 分位的个数, 随机预期≈0)。
依赖：ripser(2026-08-22 已装, ~822KB)+ numpy。长序列截断(嵌入后
点云 O(n²) 距离矩阵, 600 点足够)。
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import List, Sequence

import numpy as np

from ml.probes.surrogate_probe import make_surrogates, surrogate_zscore


@dataclass
class TdaResult:
    name: str
    metric: str            # h1_maxlife | h1_n_long
    value: float
    verdict: str           # RANDOM | NONRANDOM | INVALID
    detail: str = ""


def _embed(x: np.ndarray, dim: int = 4, tau: int = 1) -> np.ndarray:
    """时间延迟嵌入: (n-(dim-1)*tau, dim) 点云。"""
    n = x.size
    if n < dim * tau + 10:
        raise ValueError(f"序列过短 (n={n}) 无法嵌入 dim={dim}")
    idx = np.arange(n - (dim - 1) * tau)
    return np.stack([x[idx + i * tau] for i in range(dim)], axis=1)


def _persistence_summary(pc: np.ndarray, maxdim: int = 1) -> dict:
    """对点云跑 ripser, 返回 H1 环的寿命统计。

    Returns: {"h1_maxlife": float, "h1_lifetimes": list[float]}
    """
    from ripser import ripser
    res = ripser(pc, maxdim=maxdim, do_cocycles=False)
    dgms = res["dgms"]
    h1 = dgms[1] if len(dgms) > 1 else np.zeros((0, 2))
    # 寿命 = 灭 - 生; 忽略 inf(最大环)
    finite = h1[np.isfinite(h1[:, 1])]
    if finite.size == 0:
        return {"h1_maxlife": 0.0, "h1_lifetimes": []}
    lifetimes = (finite[:, 1] - finite[:, 0]).tolist()
    return {"h1_maxlife": float(max(lifetimes)), "h1_lifetimes": lifetimes}


def run_tda_probe(x: Sequence[float] | np.ndarray, dim: int = 4, tau: int = 1,
                  n_surrogates: int = 20, seed: int = 42) -> List[TdaResult]:
    """对序列跑持久同调, H1 寿命相对 RS surrogate 判显著。

    Args:
        x: 时间序列(如蓝球/和值等 i.i.d. 序列; 红球全量有组合约束伪影,
            见 visibility_probe 适用性边界)。
        dim: 嵌入维度(4 常用; 更高更贵)。
        tau: 嵌入延迟。
        n_surrogates: RS surrogate 数量(嵌入 O(n²), 不宜太大)。
    """
    arr = np.asarray(x, dtype=float).ravel()
    if arr.size < 100:
        return [TdaResult("tda_h1_maxlife", "h1_maxlife", 0.0, "INVALID", "序列过短")]
    max_points = 600  # 嵌入点云 O(n²), 截断
    if arr.size > max_points:
        arr = arr[-max_points:]
    try:
        pc = _embed(arr, dim=dim, tau=tau)
        base = _persistence_summary(pc)
    except Exception:
        return [TdaResult("tda_h1_maxlife", "h1_maxlife", 0.0, "INVALID", "嵌入/ripser 失败")]

    maxlife = base["h1_maxlife"]
    lifetimes = np.array(base["h1_lifetimes"], dtype=float)
    # 绝对阈值: 寿命超过点云中位尺度(嵌入坐标 spread) 的 1.5 倍才算"长命"。
    # (2026-08-22 修正: 原用自身 95% 分位, 随机序列也必然产生 ≈5% 长尾, 无判别力)
    spread = float(np.ptp(pc, axis=0).mean()) if pc.size else 0.0
    threshold = 1.5 * spread
    n_long = float(np.sum(lifetimes > threshold)) if lifetimes.size else 0.0

    surros = make_surrogates(arr, kind="rs", n=n_surrogates, seed=seed)

    def _stat_maxlife(s: np.ndarray) -> float:
        try:
            return _persistence_summary(_embed(s, dim, tau))["h1_maxlife"]
        except Exception:
            return float("nan")

    z = surrogate_zscore(arr, _stat_maxlife, surros)

    return [
        TdaResult("tda_h1_maxlife", "h1_maxlife", round(float(maxlife), 4),
                  "RANDOM" if abs(z) < 2 else "NONRANDOM",
                  f"z={z:.2f}; H1 环最大寿命, 随机预期短"),
        TdaResult("tda_h1_nlong", "h1_n_long", round(n_long, 0),
                  "RANDOM" if n_long <= 1 else "NONRANDOM",
                  "寿命超 95% 分位的特征数; 随机预期 0-1"),
    ]


if __name__ == "__main__":  # pragma: no cover
    rng = np.random.default_rng(7)
    demo = rng.standard_normal(600)
    for r in run_tda_probe(demo, n_surrogates=10):
        print(f"{r.name:16s} {r.metric:10s} value={r.value:8.4f} {r.verdict:8s} {r.detail}")
