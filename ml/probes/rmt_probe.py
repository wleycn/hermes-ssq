#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""RMT 随机矩阵理论 / Marchenko-Pastur 特征值谱探针 (研究简报 2026-08-20 [3])。

定位：随机性判据（高维谱家族）。首个「高维相关矩阵谱」维度：把跨球位
依赖从成对（TE/Copula）升级为整体谱结构检验。

原理（Marchenko & Pastur 1967; Laloux 1999）：
  1. 构造 N×T 数据矩阵 X：N = 33 个号码（红球全集），T = 滑动窗口期数，
     元素 X[i,t] = 号码 i 在第 t 期是否出现（0/1）；
  2. 算相关矩阵 C = corr(X)（N×N），取其特征值 {λ_i}；
  3. 理论纯噪声谱：MP 分布，支集 [λ_-, λ_+] = [(1±√q)²]，q = T/N。
     要求 T/N → ∞ 或至少 T ≫ N，q 太大时谱退化为 δ 函数（功效低）；
  4. 若存在超越随机的联合结构 → 出现超出 λ_+ 的「尖峰」特征值；
     全部落在支集内 → 纯噪声。

设计要点（2026-08-22 qwen 审核修正后规格）：
  - 用 33 号码维度（N=33），不用 6 球位（N=6 时 q≈500，MP 退化为 δ
    函数，功效极低，结论只能算「方向性证据」）；
  - N=33, T=200 时 q≈6.06；
  - 判据以 surrogate 相对显著性为准（max_ratio 的 z / spike 的经验分位）。
    注意：MP 上界 λ_+ 是渐近理论（N→∞），N=33 有限样本下 0/1 矩阵的
    max_eig 约 1.5~2.5，远达不到 λ_+≈12，故「绝对尖峰」判据在此规格下
    无检测力，spike_count 仅作信息输出，不做 NONRANDOM 依据；
  - surrogate 用 RS（逐号码洗牌，破坏跨号码同步，保留每号码出现频率）。

诚实预告：双色球 i.i.d. 均匀随机，预期 max_ratio 落在 RS-surrogate 分布
内（|z|<2）→ FLAT 证据。依赖：仅 numpy。
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Sequence

import numpy as np

from ml.probes.surrogate_probe import make_surrogates, surrogate_zscore


@dataclass
class RmtResult:
    name: str
    metric: str            # max_ratio | spike_count
    value: float
    verdict: str           # RANDOM | NONRANDOM | INVALID
    detail: str = ""


def _mp_bounds(q: float) -> tuple[float, float]:
    """MP 支集 [λ_-, λ_+]（σ²=1 归一化）。"""
    lam_min = (1 - np.sqrt(q)) ** 2
    lam_max = (1 + np.sqrt(q)) ** 2
    return float(lam_min), float(lam_max)


def build_matrix(draws: Sequence[Sequence[int]], n_numbers: int = 33,
                 window: int = 200) -> np.ndarray:
    """把开奖历史转成 N×T 出现矩阵。

    Args:
        draws: 每期红球号码列表（1-indexed），按期序排列，取最后 window 期。
        n_numbers: 号码全集大小（红球 33）。
        window: 滑动窗口期数 T。
    Returns:
        (n_numbers, T) 的 0/1 矩阵；T 不足时取实际可用期数。
    """
    reds = [sorted(d) for d in draws if len(d) >= 6]
    if len(reds) < 10:
        raise ValueError(f"开奖期数过少 (n={len(reds)}), RMT 无意义")
    recent = reds[-window:]
    T = len(recent)
    X = np.zeros((n_numbers, T), dtype=float)
    for t, row in enumerate(recent):
        for num in row[:6]:
            if 1 <= num <= n_numbers:
                X[num - 1, t] = 1.0
    return X


def rmt_spectrum(X: np.ndarray) -> Dict[str, float]:
    """计算相关矩阵特征值谱摘要。

    Returns: {"q", "lambda_min", "lambda_max", "max_eig",
              "max_ratio" (max_eig/lambda_max), "spike_count"}
    """
    N, T = X.shape
    if T <= N:
        return {"q": float("nan"), "lambda_min": float("nan"),
                "lambda_max": float("nan"), "max_eig": float("nan"),
                "max_ratio": float("nan"), "spike_count": float("nan")}
    # 每行(号码)方差为 0 的列会导致相关矩阵 NaN, 过滤
    stds = X.std(axis=1)
    valid = stds > 0
    Xv = X[valid]
    if Xv.shape[0] < 3:
        return {"q": float("nan"), "lambda_min": float("nan"),
                "lambda_max": float("nan"), "max_eig": float("nan"),
                "max_ratio": float("nan"), "spike_count": float("nan")}
    C = np.corrcoef(Xv)
    eig = np.linalg.eigvalsh(C)
    q = T / N
    lam_min, lam_max = _mp_bounds(q)
    max_eig = float(eig.max())
    # 超过 λ_+ 的 5% 容差才算尖峰(数值噪声)
    spikes = int(np.sum(eig > lam_max * 1.05))
    return {
        "q": round(float(q), 3),
        "lambda_min": round(lam_min, 4),
        "lambda_max": round(lam_max, 4),
        "max_eig": round(max_eig, 4),
        "max_ratio": round(float(max_eig / lam_max), 4),
        "spike_count": spikes,
    }


def run_rmt_probe(draws: Sequence[Sequence[int]], n_numbers: int = 33,
                  window: int = 200, n_surrogates: int = 30,
                  seed: int = 42) -> List[RmtResult]:
    """对开奖历史跑 RMT 谱检验, 接 RS surrogate 做显著性。

    判据：
      - max_ratio (最大特征值 / λ_+) 相对 surrogate 分布 |z|<2 → RANDOM；
      - spike_count 相对 surrogate 分布 (或绝对 ≤1) → RANDOM。
    预期：真实彩票两判据均落在 surrogate 分布内 → FLAT 证据。
    """
    X = build_matrix(draws, n_numbers=n_numbers, window=window)
    N, T = X.shape
    if T <= N or n_surrogates < 10:
        return [RmtResult("rmt_max", "max_ratio", 0.0, "INVALID", "窗口过短或 surrogate 过少")]

    base = rmt_spectrum(X)

    # RS surrogate: 逐号码洗牌(打乱出现时序, 保留每号码出现频率)
    rng = np.random.default_rng(seed)
    surro_ratios = []
    surro_spikes = []
    for _ in range(n_surrogates):
        Xs = np.array([rng.permutation(X[i]) for i in range(N)])
        s = rmt_spectrum(Xs)
        if np.isfinite(s["max_ratio"]):
            surro_ratios.append(s["max_ratio"])
            surro_spikes.append(s["spike_count"])

    ratios = np.array(surro_ratios)
    spikes = np.array(surro_spikes)
    if ratios.size < 5:
        return [RmtResult("rmt_max", "max_ratio", base["max_ratio"], "INVALID",
                          "surrogate 谱全部无效")]

    mr = base["max_ratio"]
    z_ratio = float((mr - ratios.mean()) / ratios.std()) if ratios.std() > 0 else 0.0
    # 尖峰数: 用 surrogate 分布经验分位(不用 z, 因计数分布偏斜)
    spike_val = float(base["spike_count"])
    frac_above = float(np.mean(spikes >= spike_val))
    p_spike = (frac_above * n_surrogates + 1) / (n_surrogates + 1)

    return [
        RmtResult("rmt_max", "max_ratio", mr,
                  "RANDOM" if abs(z_ratio) < 2 else "NONRANDOM",
                  f"z={z_ratio:.2f}, λ+/界={base['lambda_max']}, q={base['q']}"),
        RmtResult("rmt_spike", "spike_count", spike_val,
                  "RANDOM" if p_spike > 0.05 else "NONRANDOM",
                  f"p={p_spike:.3f} (surrogate 经验分位), 期望 0 尖峰"),
    ]


if __name__ == "__main__":  # pragma: no cover
    import csv
    rng = np.random.default_rng(7)
    draws = [sorted(rng.choice(33, 6, replace=False) + 1) for _ in range(400)]
    for r in run_rmt_probe(draws, n_surrogates=20):
        print(f"{r.name:12s} {r.metric:8s} value={r.value:8.4f} {r.verdict:8s} {r.detail}")
