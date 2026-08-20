#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Ordinal Pattern / Permutation Entropy 探针 (P1, 研究简报 2026-08-17 [2])。

定位：随机性判据（振幅无关、抗噪），比频谱探针更稳健的 i.i.d. 检验。
非预测器。可直接补强 ml/spectral_red.py 的探针体系（第三重正交角度）。

原理（Bandt & Pompe 2002）：
  把序列切成 d 维嵌入向量 x[i..i+d-1]，取升序 rank 得到 ordinal pattern
  （共 d! 种），统计各 pattern 经验频率 p_π，计算：
    - 排列熵 H = -Σ p_π ln p_π（归一化到 [0, ln(d!)]）；
    - Amigó χ² i.i.d. 检验：原序列 ordinal pattern 频率应≈均匀（i.i.d. 下
      每种 pattern 等概率），用 χ² 检验偏离度；显著偏离 → 非随机结构。
  - 动态可预测性监控：若某时段排列熵骤降 → 出现可辨识结构信号。

诚实预告：双色球 i.i.d. 均匀随机，预期排列熵≈ln(d!) 且 χ² 不显著 → FLAT。
依赖：仅 numpy / scipy.stats。
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, List, Sequence

import numpy as np
from scipy import stats


@dataclass
class OrdinalResult:
    name: str
    metric: str                      # permutation_entropy | chi2_iid
    value: float
    verdict: str                     # RANDOM | NONRANDOM | INVALID
    detail: str = ""


def ordinal_patterns(x: Sequence[float], d: int = 3) -> np.ndarray:
    """返回形状 (n-d+1, d) 的嵌入矩阵（每行为 d 维延迟嵌入向量）。"""
    x = np.asarray(x, dtype=float).ravel()
    n = x.size
    if n < d:
        raise ValueError(f"序列过短 n={n} < d={d}")
    return np.array([x[i:i + d] for i in range(n - d + 1)])


def permutation_entropy(x: Sequence[float], d: int = 3) -> float:
    """归一化排列熵 H/ln(d!)，∈[0,1]；随机序列→1。"""
    emb = ordinal_patterns(x, d)
    patterns = []
    for row in emb:
        # 升序排名（平局用稳定 argsort 的秩）
        rank = np.argsort(np.argsort(row, kind="stable")) + 1
        patterns.append(tuple(rank))
    # 统计各 ordinal pattern 经验频率
    from collections import Counter
    c = Counter(patterns)
    total = sum(c.values())
    probs = np.array([v / total for v in c.values()])
    H = -np.sum(probs * np.log(probs))
    return float(H / math.log(math.factorial(d)))


def amigo_chi2_iid(x: Sequence[float], d: int = 3) -> Dict:
    """Amigó χ² 检验：ordinal pattern 频率是否均匀（i.i.d. 空假设）。

    Returns: {"chi2","dof","p","verdict"}；p>0.01 → RANDOM。
    """
    emb = ordinal_patterns(x, d)
    from collections import Counter
    c = Counter(tuple(np.argsort(np.argsort(row, kind="stable")) + 1) for row in emb)
    obs = np.array(list(c.values()), dtype=float)
    expected = obs.sum() / math.factorial(d)  # 均匀期望
    # 仅保留出现过的 pattern 作 χ²（其余期望≈0 忽略）；dof = 出现种类数-1
    chi2 = np.sum((obs - expected) ** 2 / expected)
    dof = len(obs) - 1
    if dof < 1:
        return {"chi2": 0.0, "dof": 0, "p": 1.0, "verdict": "INVALID"}
    p = 1 - stats.chi2.cdf(chi2, dof)
    return {
        "chi2": round(float(chi2), 4),
        "dof": int(dof),
        "p": round(float(p), 6),
        "verdict": "RANDOM" if p > 0.01 else "NONRANDOM",
    }


def run_ordinal_probe(x: Sequence[float], dims: Sequence[int] = (3, 4, 5)) -> List[OrdinalResult]:
    """多嵌入维度跑排列熵 + Amigó χ²。"""
    x = np.asarray(x, dtype=float).ravel()
    out: List[OrdinalResult] = []
    for d in dims:
        if x.size < d + 5:
            out.append(OrdinalResult(f"pe_d{d}", "permutation_entropy", 0.0,
                                     "INVALID", "样本不足"))
            out.append(OrdinalResult(f"chi2_d{d}", "chi2_iid", 0.0,
                                     "INVALID", "样本不足"))
            continue
        pe = permutation_entropy(x, d)
        out.append(OrdinalResult(f"pe_d{d}", "permutation_entropy", round(pe, 4),
                                 "RANDOM" if pe > 0.95 else "NONRANDOM",
                                 "值越接近1越随机"))
        r = amigo_chi2_iid(x, d)
        out.append(OrdinalResult(f"chi2_d{d}", "chi2_iid", r["chi2"],
                                 r["verdict"], f"p={r['p']}, dof={r['dof']}"))
    return out
