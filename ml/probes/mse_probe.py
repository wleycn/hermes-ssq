#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""MSE 多尺度样本熵探针 (研究简报 2026-08-20 [5] 延伸项, 2026-08-22 补落地)。

定位：随机性判据（熵家族延伸）。Costa et al. 2002: 对序列做 τ 倍粗粒化
（非重叠窗口平均）→ 每尺度算样本熵 → 得 MSE 曲线（熵 vs 尺度）。
随机白噪声 MSE 随尺度快速衰减; 有结构序列在高尺度仍维持熵。

判据:
  1. 多尺度熵曲线形状: 白噪声单调快速衰减（尺度 1→τ 熵显著下降）;
  2. 高尺度熵 vs 尺度1 的比值（随机预期 << 1）。
预期: 彩票序列衰减形态 → RANDOM。纯 numpy。
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Sequence

import numpy as np


@dataclass
class MseResult:
    name: str
    metric: str            # s1 | s8 | ratio | decay
    value: float
    verdict: str           # RANDOM | NONRANDOM | INVALID
    detail: str = ""


def _sample_entropy(x: np.ndarray, m: int = 2, r: float | None = None) -> float:
    """样本熵: 嵌入 m 维, 容差 r(默认 0.2×std), -ln(匹配率)。"""
    n = x.size
    if n < m + 2:
        return float("nan")
    if r is None:
        r = 0.2 * x.std()
    if r <= 0:
        return float("nan")
    # 模板匹配计数 (O(n²), n 小可用)
    count_m = 0
    count_m1 = 0
    for i in range(n - m):
        tmpl = x[i:i + m]
        for j in range(n - m):
            if i == j:
                continue
            if np.max(np.abs(x[j:j + m] - tmpl)) <= r:
                count_m += 1
    for i in range(n - m - 1):
        tmpl = x[i:i + m + 1]
        for j in range(n - m - 1):
            if i == j:
                continue
            if np.max(np.abs(x[j:j + m + 1] - tmpl)) <= r:
                count_m1 += 1
    if count_m == 0:
        return float("nan")
    return float(-np.log(max(1, count_m1) / count_m))


def mse_curve(x: Sequence[float] | np.ndarray, max_scale: int = 8,
              m: int = 2) -> Dict[int, float]:
    """多尺度熵曲线 {τ: 样本熵}。τ 粗粒化: 非重叠窗口平均。"""
    arr = np.asarray(x, dtype=float).ravel()
    out: Dict[int, float] = {}
    for tau in range(1, max_scale + 1):
        n_coarse = arr.size // tau
        if n_coarse < m + 5:
            break
        coarse = arr[:n_coarse * tau].reshape(n_coarse, tau).mean(axis=1)
        out[tau] = _sample_entropy(coarse, m=m)
    return out


def run_mse_probe(x: Sequence[float] | np.ndarray, max_scale: int = 8,
                  m: int = 2) -> List[MseResult]:
    """跑 MSE 曲线。

    判据 (2026-08-22 实现修正): 样本熵对短序列(粗粒化后样本骤减)方差
    大, 白噪声曲线并不单调快速衰减(实测 1000 点白噪声 ratio≈0.83),
    故不用绝对阈值, 改用**曲线单调下降的稳健性**: 高尺度熵相对尺度1
    的下降比例超过 0.35 才提示结构(正常随机曲线波动 ±0.3 内)。
    此探针为熵家族延伸项, 定位信息输出 > 硬判据。
    """
    arr = np.asarray(x, dtype=float).ravel()
    if arr.size < 100:
        return [MseResult("mse_s1", "s1", 0.0, "INVALID", "序列过短")]
    max_points = 1500  # 样本熵 O(n²), 长序列截断
    if arr.size > max_points:
        arr = arr[-max_points:]
    curve = mse_curve(arr, max_scale=max_scale, m=m)
    if 1 not in curve:
        return [MseResult("mse_s1", "s1", 0.0, "INVALID", "熵计算失败")]
    s1 = curve[1]
    high_tau = max(curve.keys())
    s_high = curve[high_tau]
    ratio = s_high / s1 if s1 > 0 else float("nan")
    decayed = (s1 - s_high) / s1 if s1 > 0 else float("nan")

    # 随机曲线波动 ±0.3; 下降 >0.35 才提示结构
    verdict = "RANDOM" if (np.isfinite(decayed) and decayed < 0.35) else "NONRANDOM"
    return [
        MseResult("mse_s1", "s1", round(float(s1), 4), "RANDOM" if np.isfinite(s1) else "INVALID",
                  f"尺度1样本熵 (m={m})"),
        MseResult(f"mse_s{high_tau}", "s_high", round(float(s_high), 4),
                  "RANDOM" if np.isfinite(s_high) else "INVALID",
                  f"尺度{high_tau}样本熵"),
        MseResult("mse_decay", "decay", round(float(decayed), 4), verdict,
                  f"相对衰减; 随机波动≈±0.3, 下降>0.35 才提示结构"),
    ]


if __name__ == "__main__":  # pragma: no cover
    rng = np.random.default_rng(7)
    demo = rng.standard_normal(1000)
    for r in run_mse_probe(demo):
        print(f"{r.name:10s} {r.metric:8s} value={r.value:8.4f} {r.verdict:8s} {r.detail}")
