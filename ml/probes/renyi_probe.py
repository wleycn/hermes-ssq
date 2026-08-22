#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Rényi 广义熵谱探针 (研究简报 2026-08-22 [4], 2026-08-22 补落地)。

定位：随机性判据（熵家族延伸）。Rényi 熵 H_q = (1/(1-q)) log Σ p_i^q 是
单参数熵族: q→1 退化为 Shannon 熵, q>1 强调高频(主导)号, q<1 强调稀有号。
对号码出现频率分布扫 q 得熵谱 {q: H_q}。i.i.d. 均匀下谱形平滑且
H_q≈ln(类别数) 恒定; 分布偏离均匀(如高频号集中)时 q>1 端 H_q 显著下降。

判据:
  1. 谱形平坦度: H_{q=4} / H_{q→1} 比值(均匀预期≈1);
  2. 与理论均匀分布熵谱的距离(总变差)。
预期: 彩票号码频率≈均匀 → 谱平 → RANDOM。纯 numpy。
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import List, Sequence

import numpy as np


@dataclass
class RenyiResult:
    name: str
    metric: str            # h4_over_h1 | tv_uniform
    value: float
    verdict: str           # RANDOM | NONRANDOM | INVALID
    detail: str = ""


def renyi_entropy(probs: Sequence[float] | np.ndarray, q: float) -> float:
    """Rényi 熵 H_q = (1/(1-q)) log Σ p_i^q。q=1 用 Shannon 极限。"""
    p = np.asarray(probs, dtype=float).ravel()
    p = p[p > 0]
    if p.size == 0:
        return float("nan")
    if abs(q - 1.0) < 1e-9:
        return float(-np.sum(p * np.log(p)))
    return float(np.log(np.sum(p ** q)) / (1 - q))


def frequency_probs(values: Sequence[float] | np.ndarray, n_classes: int) -> np.ndarray:
    """把自然数序列 1..n_classes 转频率分布。"""
    arr = np.asarray(values, dtype=int).ravel()
    hist = np.bincount(arr - 1, minlength=n_classes).astype(float)
    total = hist.sum()
    return hist / total if total > 0 else np.full(n_classes, 1.0 / n_classes)


def run_renyi_probe(values: Sequence[float] | np.ndarray,
                    n_classes: int, qs: Sequence[float] = (-2, -1, 1, 2, 4)) -> List[RenyiResult]:
    """对号码频率分布跑 Rényi 熵谱检验。

    Args:
        values: 1-indexed 自然数序列(如红球全部出现值 1..33)。
        n_classes: 类别数(红 33 / 蓝 16)。
        qs: 熵阶扫描。
    """
    arr = np.asarray(values, dtype=float).ravel()
    if arr.size < 20:
        return [RenyiResult("renyi_h4_h1", "h4_over_h1", 0.0, "INVALID", "样本过少")]
    p = frequency_probs(arr, n_classes)
    uni = np.full(n_classes, 1.0 / n_classes)

    h1 = renyi_entropy(p, 1.0)
    h4 = renyi_entropy(p, 4.0)
    ratio = h4 / h1 if h1 > 0 else float("nan")
    # 与均匀分布的总变差距离
    tv = float(0.5 * np.sum(np.abs(p - uni)))
    # 均匀参考: 频率波动导致的天然 TV 阈值 ≈ 1/sqrt(n) 量级
    tv_threshold = 3.0 / np.sqrt(arr.size)

    verdict_ratio = "RANDOM" if (np.isfinite(ratio) and abs(ratio - 1.0) < 0.15) else "NONRANDOM"
    verdict_tv = "RANDOM" if tv < tv_threshold else "NONRANDOM"

    return [
        RenyiResult("renyi_h4_h1", "h4_over_h1", round(float(ratio), 4), verdict_ratio,
                    f"H4/H1; 均匀预期≈1"),
        RenyiResult("renyi_tv", "tv_uniform", round(tv, 5), verdict_tv,
                    f"vs 均匀分布总变差; 阈值≈{tv_threshold:.4f} (3/√n)"),
    ]


if __name__ == "__main__":  # pragma: no cover
    rng = np.random.default_rng(7)
    demo = rng.integers(1, 34, size=1000)
    for r in run_renyi_probe(demo, n_classes=33):
        print(f"{r.name:12s} {r.metric:10s} value={r.value:8.4f} {r.verdict:8s} {r.detail}")
