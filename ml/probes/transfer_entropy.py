#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Transfer Entropy 跨球位有向依赖检验 (P1, 研究简报 2026-08-19 [4])。

定位：唯一验证"逐球独立建模"假设是否成立的方向（现有 8 模型对每球独立建模
边际 → 均值集成）。测量球位 j 过去是否向球位 i 传递信息（有向因果）。

定义（Schreiber 2000）：
  TE(j→i) = I( X_i(t); X_j(t-1) | X_i(t-1) )
          = H(X_i(t)|X_i(t-1)) - H(X_i(t)|X_i(t-1), X_j(t-1))
  = 0 当且仅当 X_i(t) 与 X_j(t-1) 在给定 X_i(t-1) 下条件独立。

实现说明（务实、有限样本稳健）：
  - 输入：红球 6 个位各自的时间序列（或红↔蓝）。
  - 符号化：每个球位序列离散化为 k 个 bin（默认按其值域等频/等宽分箱）。
  - 用离散联合/条件频次估计熵与条件熵（加 1 平滑防 log0）。
  - 输出 TE 矩阵（i,j）+ 与 surrogate/零 TE 对照的显著性（可选）。

诚实预告：双色球 i.i.d. 均匀随机，预期所有 TE≈0（跨球位独立）→ 实证支撑
"逐球独立建模"假设；若某 TE 显著>0 才提示破坏 i.i.d. 的结构（极不可能）。
依赖：仅 numpy。
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence

import numpy as np


@dataclass
class TEResult:
    source: str
    target: str
    te: float                        # transfer entropy (nats)
    surrogate_mean: float = 0.0      # surrogate 空分布均值
    surrogate_std: float = 0.0
    z: float = 0.0                   # (te - surrogate_mean)/std
    verdict: str = "INDEPENDENT"     # INDEPENDENT | DEPENDENT | INVALID


def _symbolize(x: Sequence[float], bins: int = 8, method: str = "equal_width") -> np.ndarray:
    """把实数序列离散化为 0..bins-1 符号。"""
    x = np.asarray(x, dtype=float).ravel()
    if x.size == 0:
        return np.array([], dtype=int)
    if method == "equal_width":
        lo, hi = x.min(), x.max()
        if hi == lo:
            return np.zeros(x.size, dtype=int)
        edges = np.linspace(lo, hi, bins + 1)
        return np.clip(np.digitize(x, edges[1:-1]), 0, bins - 1)
    if method == "equal_freq":
        # 等频分箱（rank 分位）
        order = np.argsort(x)
        syms = np.empty(x.size, dtype=int)
        # 均匀切分 rank
        ranks = np.empty(x.size, dtype=int)
        ranks[order] = np.arange(x.size)
        syms = np.clip(ranks * bins // x.size, 0, bins - 1)
        return syms
    raise ValueError(f"未知 method={method}")


def _entropy(counts: np.ndarray) -> float:
    """离散频次分布的熵（nats），加 1 平滑。"""
    c = counts.astype(float) + 1.0
    p = c / c.sum()
    return float(-np.sum(p * np.log(p)))


def _cond_entropy_1d_marginal(x: np.ndarray) -> float:
    """H(X(t) | X(t-1))：相邻两时刻联合 vs 边际。"""
    a = x[:-1]
    b = x[1:]
    # 二维联合频次
    nb = int(b.max()) + 1
    na = int(a.max()) + 1
    joint = np.zeros((na, nb))
    for ai, bi in zip(a, b):
        joint[ai, bi] += 1
    h_joint = _entropy(joint)
    h_marg = _entropy(np.bincount(a, minlength=na).astype(float))
    # H(B|A) = H(A,B) - H(A)
    return h_joint - h_marg


def transfer_entropy(x_src: Sequence[float], x_tgt: Sequence[float],
                     bins: int = 8, method: str = "equal_width") -> float:
    """单方向 TE(src→tgt) = H(tgt_t | tgt_{t-1}) - H(tgt_t | tgt_{t-1}, src_{t-1})。"""
    s = _symbolize(x_src, bins, method)
    t = _symbolize(x_tgt, bins, method)
    n = min(s.size, t.size)
    s, t = s[-n:], t[-n:]
    if n < bins * 4:
        return 0.0
    # H(t_t | t_{t-1})
    h_t_given_tlag = _cond_entropy_1d_marginal(t)
    # H(t_t | t_{t-1}, s_{t-1})：三维联合 (t_{t-1}, s_{t-1}, t_t)
    a = t[:-1]          # t_{t-1}
    b = s[:-1]          # s_{t-1}
    c = t[1:]           # t_t
    nb = int(np.max([a.max(), b.max(), c.max()])) + 1
    joint = np.zeros((nb, nb, nb))
    for ai, bi, ci in zip(a, b, c):
        joint[ai, bi, ci] += 1
    h_joint3 = _entropy(joint)
    h_marg2 = _entropy(np.zeros((nb, nb)))  # placeholder; replaced below
    # H(t_{t-1}, s_{t-1})
    joint2 = np.zeros((nb, nb))
    for ai, bi in zip(a, b):
        joint2[ai, bi] += 1
    h_marg2 = _entropy(joint2)
    h_t_given_tlag_slag = h_joint3 - h_marg2
    te = h_t_given_tlag - h_t_given_tlag_slag
    return float(max(te, 0.0))  # TE 理论非负


def run_transfer_entropy_matrix(series: Dict[str, Sequence[float]],
                                bins: int = 8) -> List[TEResult]:
    """对多球位序列跑两两 TE 矩阵。

    Args:
        series: {球位名: 序列}，如 {"Red1":[...], "Red2":[...], ..., "Blue1":[...]}。
    Returns:
        TE 结果列表（含 (i,j) 与 (j,i) 双向）。
    """
    names = list(series.keys())
    out: List[TEResult] = []
    rng = np.random.default_rng(2026)
    for i in names:
        for j in names:
            if i == j:
                continue
            te = transfer_entropy(series[j], series[i], bins=bins)
            # surrogate 空分布：打乱 source 时序 N 次，估计无因果时的 TE 分布
            s_src = np.asarray(series[j], dtype=float)
            s_tgt = np.asarray(series[i], dtype=float)
            surro_tes = []
            for _ in range(50):
                s_perm = rng.permutation(s_src)
                surro_tes.append(transfer_entropy(s_perm, s_tgt, bins=bins))
            sm = float(np.mean(surro_tes))
            ss = float(np.std(surro_tes)) if np.std(surro_tes) > 0 else 1e-9
            z = (te - sm) / ss
            out.append(TEResult(
                source=j, target=i,
                te=round(te, 5),
                surrogate_mean=round(sm, 5),
                surrogate_std=round(ss, 5),
                z=round(float(z), 3),
                verdict="INDEPENDENT" if abs(z) < 2 else "DEPENDENT",
            ))
    return out


def summarize_te(results: List[TEResult]) -> dict:
    """汇总跨球位依赖：是否有任意显著 TE。"""
    n_dep = sum(1 for r in results if r.verdict == "DEPENDENT")
    n_total = len(results)
    return {
        "n_pairs": n_total,
        "n_dependent": n_dep,
        "overall": "INDEPENDENT" if n_dep == 0 else "SOME_DEPENDENCE",
        "max_te": round(max((r.te for r in results), default=0.0), 5),
        "interpretation": (
            "所有球位间 TE≈0 → 实证支撑'逐球独立建模'假设（i.i.d.）。"
            if n_dep == 0 else
            "存在显著跨球位 TE → 提示破坏 i.i.d. 的结构，需进一步排查。"
        ),
    }
