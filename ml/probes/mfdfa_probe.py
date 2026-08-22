#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""MF-DFA 多重分形去趋势波动分析探针 (研究简报 2026-08-20 [1])。

定位：随机性判据（分形-幂律家族）。现有探针全是单统计量（χ²/熵/自相关/
度分布/递归/可压缩性），本探针首次覆盖「长程记忆 / 幂律尺度 / 多重分形谱」
维度——检验序列是否存在跨尺度的长期相关性结构。

原理（Kantelhardt et al. 2002）：
  1. 对序列 x 构造 profile：Y(i) = Σ_{k=1..i} (x_k - mean(x))（积分成随机游走）；
  2. 按尺度 s 把 profile 分成 N_s 段（正反各一遍，共 2N_s 段），每段做局部
     多项式去趋势（m=1 线性），得段方差 F²(s, ν)；
  3. q 阶涨落函数 F_q(s) = [1/(2N_s) Σ F²(s,ν)^{q/2}]^{1/q}（q≠0；q=0 取
     对数平均的指数形式）；
  4. log F_q(s) vs log s 的斜率 = h(q) 广义 Hurst 指数。

i.i.d. 白噪声判据：H(q=2) ≈ 0.5 且 h(q) 为常数（单分形，谱宽 Δh≈0）。
H≠0.5 → 长程记忆；Δh>0 → 多重分形（异质尺度结构）。

诚实预告：双色球 i.i.d. 均匀随机，预期 H≈0.5、Δh≈0 → FLAT 证据。
依赖：仅 numpy（纯 numpy 实现，无 MFDFA 包依赖）。
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Sequence

import numpy as np

from ml.probes.surrogate_probe import make_surrogates, surrogate_pvalue, surrogate_zscore


@dataclass
class MfdfaResult:
    name: str
    metric: str            # h_q2 | delta_h
    value: float
    verdict: str           # RANDOM | NONRANDOM | INVALID
    detail: str = ""


def _to_float1d(x: Sequence[float] | np.ndarray) -> np.ndarray:
    arr = np.asarray(x, dtype=float).ravel()
    if arr.size < 16:
        raise ValueError(f"序列过短 (n={arr.size}), MF-DFA 至少需 16 个点")
    return arr


def _profile(x: np.ndarray) -> np.ndarray:
    """积分序列为随机游走 profile Y(i)。"""
    return np.cumsum(x - x.mean())


def _fluctuation_q(profile: np.ndarray, s: int, q: float) -> float:
    """对单一尺度 s 计算 q 阶涨落函数 F_q(s)（全向量化）。

    正反双向分段（2N_s 段）以利用整条序列；每段线性去趋势后取方差。
    线性去趋势残差方差有解析式：F² = var(y) - slope²·var(x)，避免逐段
    polyfit（x = arange(s) 固定，slope = cov(x,y)/var(x) 可矩阵化）。
    """
    n = profile.size
    nseg = n // s
    if nseg < 2:
        return float("nan")
    # 正反双向切段：形状 (2*nseg, s)
    fwd = profile[:nseg * s].reshape(nseg, s)
    rev = profile[n - nseg * s:].reshape(nseg, s)[::-1]
    segs = np.vstack([fwd, rev])                       # (2*nseg, s)
    xx = np.arange(s, dtype=float)
    x_c = xx - xx.mean()
    var_x = float(np.mean(x_c ** 2))
    if var_x == 0:
        return float("nan")
    y_c = segs - segs.mean(axis=1, keepdims=True)
    slope = (y_c @ x_c) / (var_x * s)                  # 每段斜率 (2*nseg,)
    var_y = np.mean(y_c ** 2, axis=1)
    f2 = var_y - slope ** 2 * var_x                    # 去趋势残差方差 F²
    f2 = np.clip(f2, 1e-12, None)                      # 防数值下溢
    if q == 0:
        # Kantelhardt: F_0 = exp(1/(4N_s) Σ ln F²)，N_s=单向段数，总和 2N_s 段
        return float(np.exp(0.25 / nseg * np.sum(np.log(f2))))
    return float(np.power(np.mean(f2 ** (q / 2.0)), 1.0 / q))


def h_spectrum(x: Sequence[float] | np.ndarray,
               qs: Sequence[float] = (-4, -2, 0, 2, 4),
               scales: Sequence[int] | None = None) -> Dict[float, float]:
    """计算广义 Hurst 谱 {q: h(q)}。

    Args:
        x: 原始序列（自然数编码 1..33 / 1..16 均可）。
        qs: q 阶扫描值（含 0 时为 F_0 对数平均）。
        scales: 去趋势尺度 s 列表；默认对数均匀取 8~N/4 之间 12 个。
    Returns:
        {q: h(q)}；样本不足或拟合失败时对应值记为 nan。
    """
    arr = _to_float1d(x)
    n = arr.size
    prof = _profile(arr)
    if scales is None:
        smin, smax = 8, max(9, n // 4)
        scales = np.unique(np.geomspace(smin, smax, 12).astype(int)).tolist()
    out: Dict[float, float] = {}
    for q in qs:
        fs: List[float] = []
        ls: List[float] = []
        for s in scales:
            f = _fluctuation_q(prof, s, q)
            if np.isfinite(f) and f > 0:
                fs.append(np.log(f))
                ls.append(np.log(float(s)))
        if len(fs) < 3:
            out[float(q)] = float("nan")
            continue
        # log F_q(s) ~ h(q) log s 的线性回归斜率
        slope = np.polyfit(np.array(ls), np.array(fs), 1)[0]
        out[float(q)] = float(slope)
    return out


def run_mfdfa_probe(x: Sequence[float] | np.ndarray, n_surrogates: int = 50,
                    seed: int = 42) -> List[MfdfaResult]:
    """对序列跑 MF-DFA 并用 AAFT surrogate 做显著性检验。

    判据：
      - h(q=2) 相对 surrogate 分布 |z|<2 且 ≈0.5 → RANDOM（无长程记忆）；
      - 谱宽 Δh = h(q=-4) - h(q=4) 相对 surrogate 分布 |z|<2 且 ≈0 → 单分形。
    预期：真实彩票序列两判据均落在 surrogate 分布内 → FLAT 证据。
    """
    arr = _to_float1d(x)
    if arr.size < 32 or n_surrogates < 10:
        return [MfdfaResult("mfdfa", "h_q2", 0.0, "INVALID", "样本不足或 surrogate 过少")]

    qs = (-4.0, -2.0, 0.0, 2.0, 4.0)
    h = h_spectrum(arr, qs)
    h2 = h.get(2.0, float("nan"))
    dh = h.get(-4.0, float("nan")) - h.get(4.0, float("nan"))
    if not np.isfinite(h2) or not np.isfinite(dh):
        return [MfdfaResult("mfdfa", "h_q2", float("nan"), "INVALID", "谱拟合失败")]

    # surrogate 显著性：统计量 = h(2) 和 Δh
    surros = make_surrogates(arr, kind="aaft", n=n_surrogates, seed=seed)

    def _stat_h2(s: np.ndarray) -> float:
        hs = h_spectrum(s, qs=qs)
        v = hs.get(2.0, float("nan"))
        return float("nan") if not np.isfinite(v) else float(v)

    def _stat_dh(s: np.ndarray) -> float:
        hs = h_spectrum(s, qs=qs)
        a = hs.get(-4.0, float("nan"))
        b = hs.get(4.0, float("nan"))
        if not np.isfinite(a) or not np.isfinite(b):
            return float("nan")
        return float(a - b)

    z_h2 = surrogate_zscore(arr, _stat_h2, surros)
    z_dh = surrogate_zscore(arr, _stat_dh, surros)
    p_h2 = surrogate_pvalue(arr, _stat_h2, surros)
    p_dh = surrogate_pvalue(arr, _stat_dh, surros)

    out = [
        MfdfaResult("mfdfa_h2", "h_q2", round(float(h2), 4),
                    "RANDOM" if abs(z_h2) < 2 else "NONRANDOM",
                    f"z={z_h2:.2f}, p={p_h2:.3f}; 白噪声期望 0.5"),
        MfdfaResult("mfdfa_dh", "delta_h", round(float(dh), 4),
                    "RANDOM" if abs(z_dh) < 2 else "NONRANDOM",
                    f"z={z_dh:.2f}, p={p_dh:.3f}; 单分形期望 0"),
    ]
    return out


if __name__ == "__main__":  # pragma: no cover
    rng = np.random.default_rng(7)
    demo = rng.integers(1, 34, size=500).astype(float)
    for r in run_mfdfa_probe(demo, n_surrogates=50):
        print(f"{r.name:12s} {r.metric:8s} value={r.value:8.4f} {r.verdict:8s} {r.detail}")
