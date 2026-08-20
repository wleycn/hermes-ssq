#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Surrogate Data Testing 元验证框架 (P0, 研究简报 2026-08-18 [4])。

定位：所有探针（频谱 / ordinal / 可见图 / RQA / LZ / NIST）的**对抗性验证环**。
本身不是预测器，而是检验"我们的探针是否真能区分随机 vs 结构"的金标准反证法。

原理（Theiler et al. 1992 "Testing for nonlinearity"）：
  对原序列 x 生成 N 组 surrogate，各 surrogate 保持某个**空假设**特征不变，
  但打乱它想检测的结构。再用任意非线性统计量 T 对比：
    - 若 T(x) 落在 surrogate 的 T 分布内 → 无法拒绝"x 满足该空假设"，判 RANDOM；
    - 若 T(x) 显著偏离（|z| 大）→ 存在超出空假设的结构。

三种 surrogate（涵盖主要空假设族）：
  1. RS  (Random Shuffle)：完全打乱顺序，保留振幅分布，破坏一切时序结构。
  2. AAFT (Amplitude Adjusted Fourier Transform)：相位随机化（FFT 后随机化相位、
       逆变换），保留功率谱（线性相关性）但破坏非线性/相位结构。
  3. IAAFT (Iterative AAFT)：迭代逼近，使 surrogate 同时保留原序列的振幅分布
       与功率谱，更严格的线性随机过程空假设。

用法：
  from ml.probes.surrogate_probe import make_surrogates, surrogate_zscore
  surros = make_surrogates(red_series, kind="aaft", n=1000, rng=42)
  z = surrogate_zscore(red_series, np.mean, surros)   # z<2 即无法拒绝空假设

诚实预告：双色球为 i.i.d. 均匀随机，预期全部 |z|<2 → RANDOM。这是科学结论，
不是失败；本框架的意义在于把"FLAT"从单探针不显著升级为"在反证法下稳健不显著"。
依赖：仅 numpy（FFT 用 numpy.fft）。
"""
from __future__ import annotations

import numpy as np
from typing import Callable, List, Sequence


def _to_float1d(x: Sequence[float]) -> np.ndarray:
    arr = np.asarray(x, dtype=float).ravel()
    if arr.size < 4:
        raise ValueError(f"序列过短 (n={arr.size}), surrogate 无意义")
    return arr


def make_surrogates(
    x: Sequence[float],
    kind: str = "aaft",
    n: int = 1000,
    seed: int = 42,
) -> List[np.ndarray]:
    """生成 n 组 surrogate 序列。

    Args:
        x: 原序列（任意实数；对彩票可传自然数编码 1..33 / 1..16）。
        kind: "rs" | "aaft" | "iaaft"。
        n: surrogate 数量。
        seed: RNG 种子（可复现）。
    Returns:
        list[np.ndarray]，每个长度同 x。
    """
    x = _to_float1d(x)
    rng = np.random.default_rng(seed)
    if kind == "rs":
        return [_rs_surrogate(x, rng) for _ in range(n)]
    if kind == "aaft":
        return [_aaft_surrogate(x, rng) for _ in range(n)]
    if kind == "iaaft":
        return [_iaaft_surrogate(x, rng) for _ in range(n)]
    raise ValueError(f"未知 kind={kind!r}; 用 'rs'/'aaft'/'iaaft'")  # noqa: B904


def _rs_surrogate(x: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Random Shuffle：保留值分布，完全破坏时序。"""
    return rng.permutation(x).astype(float)


def _aaft_surrogate(x: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Amplitude Adjusted Fourier Transform（单次，非迭代）。

    步骤：1) 对 x 排序得 rank 索引；2) 生成高斯白噪声并随机化其相位（FFT）；
    3) 把高斯序列排序后按 x 的 rank 索引重排，得到保留 x 振幅分布 +
    近似保留 x 功率谱的 surrogate。
    """
    n = x.size
    # 原序列的排序顺序（rank 索引）：让 surrogate 在排序后能被 x 的排序映射
    order = np.argsort(x)
    # 高斯白噪声
    g = rng.standard_normal(n)
    # 相位随机化：FFT -> 随机化相位 -> IFFT
    G = np.fft.rfft(g)
    phases = rng.uniform(0, 2 * np.pi, G.shape[0])
    G_rand = np.abs(G) * np.exp(1j * phases)
    g_sur = np.fft.irfft(G_rand, n=n)
    # 把 x 的值按 g_sur 的排序位置重新映射（保留 x 振幅分布）
    g_order = np.argsort(g_sur)
    s = np.empty(n, dtype=float)
    s[g_order] = x[order]
    return s


def _iaaft_surrogate(x: np.ndarray, rng: np.random.Generator, iters: int = 10) -> np.ndarray:
    """Iterative AAFT：迭代收敛到同时匹配振幅分布与功率谱的 surrogate。"""
    n = x.size
    order = np.argsort(x)
    # 初始 AAFT
    s = _aaft_surrogate(x, rng)
    target_fft = np.fft.rfft(x)
    target_amp = np.abs(target_fft)
    for _ in range(iters):
        # 强制匹配功率谱：用 s 的相位 + x 的幅值
        S = np.fft.rfft(s)
        S_new = target_amp * np.exp(1j * np.angle(S))
        s_new = np.fft.irfft(S_new, n=n)
        # 强制匹配振幅分布（排序重排）
        g_order = np.argsort(s_new)
        s = np.empty(n, dtype=float)
        s[g_order] = x[order]
    return s


def surrogate_zscore(
    x: Sequence[float],
    stat: Callable[[np.ndarray], float],
    surrogates: Sequence[np.ndarray],
) -> float:
    """计算原序列统计量相对 surrogate 分布的 z-score。

    z = (T(x) - mean(T(surros))) / std(T(surros))
    |z| < 2 通常视为无法拒绝空假设（无结构）。
    """
    tx = float(stat(_to_float1d(x)))
    ts = np.array([float(stat(np.asarray(s, dtype=float))) for s in surrogates])
    sd = ts.std(ddof=0)
    if sd == 0:
        return 0.0
    return (tx - ts.mean()) / sd


def surrogate_pvalue(
    x: Sequence[float],
    stat: Callable[[np.ndarray], float],
    surrogates: Sequence[np.ndarray],
    two_sided: bool = True,
) -> float:
    """基于 surrogate 分布的经验 p 值（更稳健，不假设正态）。

    返回原序列统计量在 surrogate 分布中的分位尾部概率。
    """
    tx = float(stat(_to_float1d(x)))
    ts = np.array([float(stat(np.asarray(s, dtype=float))) for s in surrogates])
    n = ts.size
    if two_sided:
        # 双侧：偏离均值方向两尾合计
        mean_t = ts.mean()
        extreme = np.sum(np.abs(ts - mean_t) >= np.abs(tx - mean_t))
    else:
        extreme = np.sum(ts >= tx)
    # +1 平滑，避免 p=0
    return (extreme + 1) / (n + 1)


def run_surrogate_probe(
    x: Sequence[float],
    stats: dict[str, Callable[[np.ndarray], float]],
    kinds: Sequence[str] = ("rs", "aaft", "iaaft"),
    n: int = 1000,
    seed: int = 42,
) -> dict:
    """对序列 x 跑多统计量 × 多种 surrogate 的元验证。

    Returns: {stat_name: {kind: {"z": float, "p": float, "verdict": "RANDOM"|"STRUCTURE"}}}
    """
    x = _to_float1d(x)
    out: dict = {}
    for sname, stat in stats.items():
        out[sname] = {}
        for kind in kinds:
            surros = make_surrogates(x, kind=kind, n=n, seed=seed)
            z = surrogate_zscore(x, stat, surros)
            p = surrogate_pvalue(x, stat, surros)
            out[sname][kind] = {
                "z": round(float(z), 4),
                "p": round(float(p), 4),
                "verdict": "RANDOM" if abs(z) < 2 else "STRUCTURE",
            }
    return out
