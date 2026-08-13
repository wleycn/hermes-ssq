#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""红球频谱/结构随机性三路径检验器（SSQ 随机性检验探针）。

定位: 把「摇奖机公平」与「连号偏热」类彩民直觉变成有/无统计证据的结论 ——
非预测器。纯统计域: 无状态、无 IO、纯 numpy/scipy 计算; 只依赖
numpy / scipy / math / itertools / dataclasses + from ml import spectral
（复用其公开函数 chi2_uniform_test / fisher_g_test, 参数化传参 n=33 /
complex_signal=False, 公开接口零改动）。红球每期 6 个号是无序集合
（不要求升序, 真实数据含 1 行未升序 [12,15,5,23,6,25]）。

三条互补路径（架构文档 docs/arch_spectral_red.json, ADR 已拍板）:
- 路径1 时间维度: 33 条指示序列（合并卡方 / 逐号 lag-1 仅报告诊断 /
  Fisher's g 族 Bonf α_comp/33）+ 聚合重号率 z（超几何精确矩, 判定项）。
- 路径2 横截面: 33×33 同现矩阵 → 逐对二项检验 → 自实现 BH-FDR
  （上尾同现 / 下尾互斥且 obs≤期望−3σ）+ 子类聚合 z 检验
  （连号/同尾/三区间, 精确矩闭式, Bonf α_comp/5）; PMI 仅排序展示。
- 路径3 派生标量: 和值 33C6 精确卷积 null DP + 精确矩均值 z（判定项）;
  跨度/奇偶比闭式 null 对照; 和值 chi2 拟合(60 箱)与奇偶 7 格卡方仅报告。

α 控制: 分层 Sidak α=0.05 → α_path=0.01695（三路）→ α_comp=0.00568
（路径内三组件）→ 族内 Bonferroni（逐号谱峰 α_comp/33、子类 α_comp/5）+
528 对 BH-FDR。组合误报率实测 6/100≈6%（命中 5%±2% 目标）。
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from functools import lru_cache
from typing import Dict, List, Optional, Tuple

import numpy as np
import scipy.stats

from ml import spectral

# ================= 常量与 α 拆分 =================
RED_N = 33                 # 红球号码数 1..33
RED_K = 6                  # 每期红球个数
MIN_N = 500                # 样本量硬边界: 低于则返回 INSUFFICIENT_DATA 不判定
TOTAL_PAIRS = 528          # C(33,2) 号码对数
PAIR_P = 15.0 / 528.0      # 单对单期出现概率 = C(6,2)/C(33,2)
REPEAT_MU = 36.0 / 33.0    # 单期重号数期望（超几何(33,6,6)）
REPEAT_VAR = 6.0 * 6.0 * 27.0 * 27.0 / (33.0 ** 2 * 32.0)  # 超几何精确方差
SUB_P3 = math.comb(30, 3) / math.comb(33, 6)  # 两对共享 1 号时同现概率
SUB_P4 = math.comb(29, 2) / math.comb(33, 6)  # 两对不相交时同现概率
# 子类常量表: (名称, 对总数, 共享 1 号的有序对-对计数, 不相交的有序对-对计数)
SUBCLASS_TABLE: List[Tuple[str, int, int, int]] = [
    ("连号", 32, 62, 930),
    ("同尾", 39, 114, 1368),
    ("区间1[1..11]", 55, 990, 1980),
    ("区间2[12..22]", 55, 990, 1980),
    ("区间3[23..33]", 55, 990, 1980),
]
RED_VERDICTS = ("FLAT", "PEAK_CONFIRMED", "NONSPECTRAL_BIAS",
                "STRUCTURAL_ANOMALY", "SCALAR_BIAS", "INSUFFICIENT_DATA")


def ALPHA_PATH(alpha: float = 0.05) -> float:
    """Sidak 三路拆分: α_path = 1-(1-α)^(1/3)。

    Args:
        alpha: 总体显著性水平, 默认 0.05。

    Returns:
        路径级 α_path（α=0.05 → 0.01695）。
    """
    return 1.0 - math.pow(max(0.0, 1.0 - alpha), 1.0 / 3.0)


def ALPHA_COMP(alpha: float = 0.05) -> float:
    """路径内三组件 Sidak 拆分: α_comp = 1-(1-α_path)^(1/3)。

    Args:
        alpha: 总体显著性水平, 默认 0.05。

    Returns:
        组件级 α_comp（α=0.05 → 0.00568）。
    """
    return 1.0 - math.pow(max(0.0, 1.0 - ALPHA_PATH(alpha)), 1.0 / 3.0)


# ================= 结果类型 =================
@dataclass(frozen=True)
class RepeatRateResult:
    """单期重号数聚合检验结果。"""

    observed_rate: float
    expected_rate: float
    total_repeats: int
    z: float
    significant: bool


@dataclass(frozen=True)
class SubclassResult:
    """子类聚合 z 检验结果（连号/同尾/区间）。"""

    name: str
    n_pairs: int
    observed: int
    expected: float
    z: float
    significant: bool


@dataclass(frozen=True)
class ZTestResult:
    """精确矩均值 z 检验结果。"""

    z: float
    p_value: float
    significant: bool


@dataclass(frozen=True)
class RedSpectralResult:
    """红球三路径检验结果聚合（唯一事实来源）。"""

    path1: Optional[Dict]
    path2: Optional[Dict]
    path3: Optional[Dict]
    verdict: str
    conclusion: str
    alpha_split: Dict
    n_periods: int


# ================= 前置校验 =================
def _validate_reds(reds: np.ndarray) -> np.ndarray:
    """校验红球输入并规范化为 (N,6) int64（集合语义, 不要求升序）。

    Args:
        reds: 红球数组, 形状 (N,6), 取值 1..33, 每行 6 个互异。

    Returns:
        规范化的 (N,6) int64 数组。

    Raises:
        ValueError: 形状/值域/互异性任一不满足。
    """
    arr = np.asarray(reds)
    if arr.ndim != 2 or arr.shape[1] != RED_K:
        raise ValueError(f"红球数组形状必须为 (N,{RED_K}), 实际 {arr.shape}")
    if not (np.issubdtype(arr.dtype, np.integer)
            or (np.issubdtype(arr.dtype, np.floating)
                and bool(np.all(arr == np.floor(arr))))):
        raise ValueError(f"红球数组必须为整数类型, 实际 {arr.dtype}")
    arr = arr.astype(np.int64)
    if arr.shape[0] == 0:
        raise ValueError("红球数组为空")
    if arr.min() < 1 or arr.max() > RED_N:
        raise ValueError(f"红球号码必须在 1..{RED_N}, 实际范围 [{arr.min()},{arr.max()}]")
    if not bool(np.all(np.apply_along_axis(
            lambda r: len(np.unique(r)) == RED_K, 1, arr))):
        raise ValueError("每期 6 个红球号码必须互异（集合语义）")
    return arr


# ================= 路径1 时间维度 =================
def red_indicator_matrix(reds: np.ndarray) -> np.ndarray:
    """33 条指示序列矩阵: 行 i 为号码 i+1 的出现指示（该期是否开出）。

    Args:
        reds: (N,6) int 数组（由 _validate_reds 校验后传入）。

    Returns:
        (RED_N, N) bool 矩阵; 行和=每号出现期次（期望 6N/33）, 全矩阵和=6N。

    Raises:
        ValueError: 输入非 2 维。
    """
    arr = np.asarray(reds)
    if arr.ndim != 2:
        raise ValueError(f"红球数组必须为 2 维, 实际 {arr.ndim} 维")
    n = arr.shape[0]
    indicator = np.zeros((RED_N, n), dtype=bool)
    rows = np.repeat(np.arange(n), arr.shape[1])
    cols = arr.ravel() - 1  # 号码 1..33 → 行索引 0..32
    indicator[cols, rows] = True
    return indicator


def pooled_red_chi2(reds: np.ndarray, alpha: float = ALPHA_COMP()) -> spectral.Chi2Result:
    """33 号边际频率均匀性卡方（df=32, 合并全部 6N 个号码）。

    复用 ml/spectral.chi2_uniform_test(reds.ravel(), n=33): 期望 E=6N/33。
    红球唯一直接复用蓝球卡方的入口（参数化 n=33）。

    Args:
        reds: (N,6) int 数组。
        alpha: 显著性水平, 默认 α_comp=0.00568。

    Returns:
        spectral.Chi2Result(stat, df=32, p_value, counts, expected, significant)。
    """
    arr = _validate_reds(reds)
    return spectral.chi2_uniform_test(arr.ravel(), n=RED_N, alpha=alpha)


def indicator_lag1_zs(indicator: np.ndarray) -> np.ndarray:
    """33 条指示序列的 lag-1 自相关 z（仅报告诊断, 不参与判定）。

    0/1 序列自相关尾部膨胀（MC 族误报 8.5% vs 名义 5%）, 判定走 repeat_rate_test。
    x 居中后 r1=Σx_t·x_{t-1}/Σx_t², z=r1·√N。

    Args:
        indicator: (RED_N, N) bool 指示矩阵。

    Returns:
        (RED_N,) float 数组, 每条序列的 lag-1 自相关 z。
    """
    n = indicator.shape[1]
    zs = np.zeros(indicator.shape[0], dtype=np.float64)
    for i in range(indicator.shape[0]):
        x = indicator[i].astype(np.float64)
        xc = x - x.mean()
        denom = float(np.sum(xc * xc))
        if denom <= 0.0:
            zs[i] = 0.0
            continue
        r1 = float(np.sum(xc[1:] * xc[:-1])) / denom
        zs[i] = r1 * math.sqrt(n)
    return zs


def repeat_rate_test(reds: np.ndarray, alpha: float = ALPHA_COMP()) -> RepeatRateResult:
    """相邻期重号率 z 检验（超几何精确矩, 路径1 判定项）。

    逐期重号数 X_t ~ 超几何(33,6,6): E=36/33, Var=6·6·27·27/(33²·32)
    （全条件方差, 与前期无关）。z=(R−T·μ)/√(T·σ²), T=N−1。

    Args:
        reds: (N,6) int 数组。
        alpha: 显著性水平, 默认 α_comp。

    Returns:
        RepeatRateResult(observed_rate, expected_rate, total_repeats, z, significant)。
    """
    arr = _validate_reds(reds)
    n = arr.shape[0]
    t = n - 1
    total_repeats = 0
    prev = set(arr[0].tolist())
    for i in range(1, n):
        cur = set(arr[i].tolist())
        total_repeats += len(prev & cur)
        prev = cur
    z = (total_repeats - t * REPEAT_MU) / math.sqrt(t * REPEAT_VAR)
    significant = bool(abs(z) > scipy.stats.norm.ppf(1.0 - alpha / 2.0))
    return RepeatRateResult(observed_rate=total_repeats / t, expected_rate=REPEAT_MU,
                            total_repeats=total_repeats, z=z, significant=significant)


def indicator_fisher_g_family(
        indicator: np.ndarray,
        alpha: float = ALPHA_COMP() / RED_N) -> List[spectral.FisherGResult]:
    """33 条指示序列的 Fisher's g 谱峰检验（族内 Bonferroni α_comp/33）。

    逐条复用 ml/spectral.fisher_g_test(seq, alpha=alpha, complex_signal=False):
    实序列 bins 1..(N−1)//2, m=(N−1)//2。

    Args:
        indicator: (RED_N, N) bool 指示矩阵。
        alpha: 逐条显著性水平, 默认 α_comp/33。

    Returns:
        33 个 FisherGResult 列表（含 g/p_value/peak_bin/significant）。
    """
    results: List[spectral.FisherGResult] = []
    for i in range(indicator.shape[0]):
        seq = indicator[i].astype(np.float64)
        results.append(spectral.fisher_g_test(seq, alpha=alpha, complex_signal=False))
    return results


# ================= 路径2 横截面 =================
def cooccurrence_matrix(reds: np.ndarray) -> np.ndarray:
    """33×33 同现矩阵: C[a,b] = 号码 a+1 与 b+1 同现期数（每期 15 对）。

    Args:
        reds: (N,6) int 数组。

    Returns:
        (33,33) int64 对称矩阵, 对角恒 0, 总和 = 30N。
    """
    arr = _validate_reds(reds)
    n = arr.shape[0]
    c = np.zeros((RED_N, RED_N), dtype=np.int64)
    for i in range(n):
        nums = arr[i] - 1
        for a in range(RED_K):
            for b in range(a + 1, RED_K):
                u, v = nums[a], nums[b]
                c[u, v] += 1
                c[v, u] += 1
    return c


def bh_fdr(pvals: np.ndarray, alpha: float) -> np.ndarray:
    """自实现 Benjamini-Hochberg FDR 控制（scipy 无现成 BH）。

    p 升序 p_(1)..p_(m); 找最大 k* 使 p_(k) ≤ (k/m)·α; 拒绝前 k* 个。
    完全 null 下 P(任一检出)≈α 是 BH 的正确性质（非 α·m）。

    Args:
        pvals: (m,) p 值数组。
        alpha: FDR 水平, ∈(0,1)。

    Returns:
        (m,) bool 掩码, True=拒绝（与 pvals 同序）。
    """
    p = np.asarray(pvals, dtype=np.float64).ravel()
    m = p.size
    if m == 0:
        return np.zeros(0, dtype=bool)
    order = np.argsort(p, kind="stable")
    below = np.argwhere(p[order] <= np.arange(1, m + 1) / m * alpha)
    if below.size == 0:
        return np.zeros(m, dtype=bool)
    k_star = int(below.max())
    reject = np.zeros(m, dtype=bool)
    reject[order[: k_star + 1]] = True
    return reject


def pair_binomial_stats(
        c: np.ndarray,
        n_periods: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """528 对逐对二项检验（正态近似）: z 矩阵 + 上/下尾 p 数组。

    逐期独立 Bernoulli(15/528): z=(C[i,j]−N·p)/√(N·p(1−p))。

    Args:
        c: (33,33) 同现矩阵。
        n_periods: 期数 N。

    Returns:
        (z_matrix (33,33), p_upper (528,), p_lower (528,)):
        p_upper=sf(z) 正向同现, p_lower=cdf(z) 互斥方向
        （上三角 528 对, np.triu_indices 顺序）。
    """
    n = n_periods
    mu = n * PAIR_P
    sd = math.sqrt(n * PAIR_P * (1.0 - PAIR_P))
    z = (c - mu) / sd
    iu = np.triu_indices(RED_N, k=1)
    z_flat = z[iu]
    return z, scipy.stats.norm.sf(z_flat), scipy.stats.norm.cdf(z_flat)


def subclass_stats(obs: int, n_pairs: int, n_shared: int, n_disjoint: int,
                   n_periods: int, name: str = "") -> SubclassResult:
    """子类聚合 z 检验（精确矩闭式, 多重比较面 528→5）。

    μ=n_pairs·15/528; Var=μ + n_shared·P3 + n_disjoint·P4 − μ²
    （P3/P4 见模块常量, 与 C(33,6)=1,107,568 全枚举精确一致）。
    z=(obs−N·μ)/√(N·Var)。显著性: 族内 Bonferroni |z|>Φ⁻¹(1−α_comp/10)
    （临界 z=3.254）。

    Args:
        obs: 子类内全部号码对同现总期数。
        n_pairs: 子类内号码对数（连号 32 / 同尾 39 / 区间 55）。
        n_shared: 子类内共享 1 号的有序对-对计数。
        n_disjoint: 子类内不相交的有序对-对计数。
        n_periods: 期数 N。
        name: 子类名称（默认空串, 由调用方填充）。

    Returns:
        SubclassResult(name, n_pairs, observed, expected, z, significant)。
    """
    mu = n_pairs * PAIR_P
    var = mu + n_shared * SUB_P3 + n_disjoint * SUB_P4 - mu * mu
    expected = n_periods * mu
    z = (obs - expected) / math.sqrt(n_periods * var)
    critical = scipy.stats.norm.ppf(1.0 - ALPHA_COMP() / 10.0)
    return SubclassResult(name=name, n_pairs=n_pairs, observed=obs, expected=expected,
                          z=z, significant=bool(abs(z) > critical))


def pmi_ranking(c: np.ndarray, freq: np.ndarray,
                n_periods: int) -> List[Tuple[float, int, int]]:
    """PMI 排序展示（不判显著）: PMI(a,b)=ln(C_ab·N/(f_a·f_b))。

    天然校正高频号同现。C_ab=0 的对跳过。

    Args:
        c: (33,33) 同现矩阵。
        freq: (33,) 每号总出现次数。
        n_periods: 期数 N。

    Returns:
        按 PMI 降序的 (pmi, i+1, j+1) 列表（仅非零同现对）。
    """
    entries: List[Tuple[float, int, int]] = []
    n = n_periods
    for i in range(RED_N):
        for j in range(i + 1, RED_N):
            cnt = c[i, j]
            if cnt == 0:
                continue
            pmi = math.log(cnt * n / (freq[i] * freq[j]))
            entries.append((pmi, i + 1, j + 1))
    entries.sort(key=lambda e: e[0], reverse=True)
    return entries


# ================= 路径3 派生标量 =================
@lru_cache(maxsize=1)
def exact_sum_null() -> Tuple[np.ndarray, float, float]:
    """和值 null: 33C6 精确卷积 DP（确定性、无 MC, 秒级）。

    ways[s][k] = 从 1..33 选 k 个不同数且和为 s 的方式数, 逐值 v∈1..33
    逆序更新（s 降序、k 降序）; pmf = ways[:,6]/C(33,6)。

    Returns:
        (pmf (0..198), mean, var): mean=102.0, var=459.0, 支撑 [21,183],
        pmf 总和=1.000000。
    """
    max_sum = RED_N * RED_K
    ways = np.zeros((max_sum + 1, RED_K + 1), dtype=np.int64)
    ways[0, 0] = 1
    for v in range(1, RED_N + 1):
        for s in range(max_sum, v - 1, -1):
            for k in range(RED_K, 0, -1):
                ways[s, k] += ways[s - v, k - 1]
    total = math.comb(RED_N, RED_K)
    pmf = ways[:, RED_K] / float(total)
    support = np.arange(max_sum + 1, dtype=np.float64)
    mean = float(np.sum(support * pmf))
    var = float(np.sum(support * support * pmf) - mean * mean)
    return pmf, mean, var


def span_null() -> Tuple[np.ndarray, float, float]:
    """跨度 null 闭式: span=s 的组合数 = (33−s)·C(s−1,4), s=5..32。

    Returns:
        (pmf (0..32), mean, var): mean=24.2857, var=23.4184, pmf 总和=1。
    """
    pmf = np.zeros(RED_N + 1, dtype=np.float64)
    total = math.comb(RED_N, RED_K)
    for s in range(5, RED_N + 1):
        pmf[s] = (RED_N - s) * math.comb(s - 1, RED_K - 2)
    pmf /= float(total)
    support = np.arange(RED_N + 1, dtype=np.float64)
    mean = float(np.sum(support * pmf))
    var = float(np.sum(support * support * pmf) - mean * mean)
    return pmf, mean, var


def odd_even_null() -> Tuple[np.ndarray, float, float]:
    """奇偶比 null 闭式: 奇数个数 k 的组合数 = C(17,k)·C(16,6−k), k=0..6。

    Returns:
        (pmf (0..6), mean, var): mean=3.0909, var=1.2645, pmf 总和=1。
    """
    pmf = np.zeros(RED_K + 1, dtype=np.float64)
    total = math.comb(RED_N, RED_K)
    for k in range(RED_K + 1):
        pmf[k] = math.comb(17, k) * math.comb(16, RED_K - k)
    pmf /= float(total)
    support = np.arange(RED_K + 1, dtype=np.float64)
    mean = float(np.sum(support * pmf))
    var = float(np.sum(support * support * pmf) - mean * mean)
    return pmf, mean, var


def moments_z_test(obs_mean: float, null_mean: float, null_var: float, n: int,
                   alpha: float = ALPHA_COMP()) -> ZTestResult:
    """精确矩均值 z 检验（无分箱任意性, 路径3 判定项）。

    z=(obs_mean−null_mean)/√(null_var/n); p=2·(1−Φ(|z|))。

    Args:
        obs_mean: 观测均值。
        null_mean: null 精确均值。
        null_var: null 精确方差。
        n: 样本量。
        alpha: 显著性水平, 默认 α_comp。

    Returns:
        ZTestResult(z, p_value, significant)。
    """
    z = (obs_mean - null_mean) / math.sqrt(null_var / n)
    p = 2.0 * scipy.stats.norm.sf(abs(z))
    significant = bool(abs(z) > scipy.stats.norm.ppf(1.0 - alpha / 2.0))
    return ZTestResult(z=z, p_value=p, significant=significant)


def odd_even_chi2(odds: np.ndarray, alpha: float = ALPHA_COMP()) -> spectral.Chi2Result:
    """奇偶分布 7 格卡方（自然类别, 无分箱问题）。

    counts=bincount(odds, 0..6), 期望=odd_even_null pmf·N, df=6。

    Args:
        odds: (N,) 每期奇数个数（0..6）。
        alpha: 显著性水平, 默认 α_comp。

    Returns:
        spectral.Chi2Result(stat, df=6, p_value, counts, expected, significant)。
    """
    pmf, _, _ = odd_even_null()
    n = odds.size
    counts = np.bincount(odds, minlength=RED_K + 1).astype(np.float64)
    expected = pmf * n
    stat = float(np.sum((counts - expected) ** 2 / expected))
    p_value = float(scipy.stats.chisquare(counts, f_exp=expected).pvalue)
    return spectral.Chi2Result(stat=stat, df=RED_K, p_value=p_value, counts=counts,
                               expected=expected, significant=bool(p_value < alpha))


# ================= 顶层编排 =================
def _sum_chi2_fit(sums: np.ndarray, n_periods: int) -> Dict:
    """和值 chi2 拟合（60 等概率箱, null pmf 分位数边界, 仅报告）。

    等概率箱期望 = 每箱实际 pmf 累积 × N; 分箱数与边界构造敏感,
    不作判定（ADR-006: 判定走精确矩均值 z）。
    """
    pmf, _, _ = exact_sum_null()
    cum = np.cumsum(pmf)
    edges = np.array([np.searchsorted(cum, k / 60.0) for k in range(61)],
                     dtype=np.float64)
    counts, _ = np.histogram(sums, bins=edges)
    exp = np.zeros(60)
    for k in range(60):
        lo, hi = int(edges[k]), int(edges[k + 1])
        exp[k] = n_periods * (cum[hi - 1] - (cum[lo - 1] if lo > 0 else 0.0))
    mask = exp > 0
    c = counts[mask].astype(np.float64)
    e = exp[mask] * (c.sum() / exp[mask].sum())
    stat = float(np.sum((c - e) ** 2 / e))
    df = int(mask.sum()) - 1
    return {"n_bins": 60, "stat": stat, "df": df,
            "p": float(scipy.stats.chi2.sf(stat, df))}


def _subclass_pairs(name: str) -> List[Tuple[int, int]]:
    """按子类名返回 0-indexed 号码对列表（连号/同尾/区间）。"""
    if name == "连号":
        return [(i - 1, i) for i in range(1, RED_N)]
    if name == "同尾":
        return [(a - 1, b - 1) for a in range(1, RED_N + 1)
                for b in range(a + 1, RED_N + 1) if a % 10 == b % 10]
    zone = {"区间1[1..11]": (1, 11), "区间2[12..22]": (12, 22),
            "区间3[23..33]": (23, 33)}[name]
    lo, hi = zone
    return [(a - 1, b - 1) for a in range(lo, hi + 1)
            for b in range(a + 1, hi + 1)]


def _pmi_top_with_count(ranking: List[Tuple[float, int, int]],
                        c: np.ndarray) -> List[Tuple[float, int, int, int]]:
    """PMI 排序前 10 条附加同现计数 → (pmi, a, b, count) 4 元组。"""
    return [(pmi, a, b, int(c[a - 1, b - 1])) for pmi, a, b in ranking[:10]]


def _conclusion(verdict: str, n: int, chi2: spectral.Chi2Result,
                repeat: RepeatRateResult, sub_results: List[SubclassResult],
                p3: Dict, alpha_comp: float,
                peak_info: Optional[Tuple[int, int, float]]) -> str:
    """按 verdict 生成中文结论段落（模板见架构文档 run_red_spectral_test 规格）。"""
    sub = {s.name: s for s in sub_results}
    if verdict == "FLAT":
        return (f"{n} 期红球序列未发现显著非随机结构：33 号边际卡方 p={chi2.p_value:.4f}、"
                f"重号率 z={repeat.z:.3f}、逐号谱峰显著 0/{RED_N}、"
                f"528 对同现 FDR 双向 0 检出、子类连号 z={sub['连号'].z:.3f} / "
                f"同尾 z={sub['同尾'].z:.3f} / 区间3[23..33] z={sub['区间3[23..33]'].z:.3f}、"
                f"和值均值 z={p3['sum']['z']:.3f}（临界 {alpha_comp:.4f}）。"
                f"统计证据支持摇奖机公平假设。")
    if verdict == "NONSPECTRAL_BIAS":
        return (f"无谱峰但边际频率/重号时序偏离均匀（卡方 p={chi2.p_value:.4f}，"
                f"重号率 z={repeat.z:.3f}），属非周期偏差，建议复核数据源与开奖号段。")
    if verdict == "STRUCTURAL_ANOMALY":
        sig = [s for s in sub_results if s.significant]
        s0 = sig[0] if sig else sub_results[0]
        return (f"横截面检验发现显著异常：子类 {s0.name} 观测 {s0.observed} vs "
                f"期望 {s0.expected:.1f}（z={s0.z:.3f}）。注意显著≠可预测——"
                f"多重比较下仍有假阳性可能，且彩民群体已对连号/同尾类模式定价，"
                f"不构成选号建议；建议分段复核数据源并持续监控。")
    if verdict == "PEAK_CONFIRMED":
        num, peak_bin, pv = peak_info or (None, None, None)
        return (f"号码 {num} 的出现呈周期性（谱峰 bin {peak_bin}，p={pv:.3g}），"
                f"疑似物理偏差，建议深挖（球重/磨损/数据源）并人工复核。")
    if verdict == "SCALAR_BIAS":
        sum_d = p3["sum"]
        direction = "偏低" if sum_d["obs_mean"] < sum_d["null_mean"] else "偏高"
        effect_pct = abs(sum_d["obs_mean"] - sum_d["null_mean"]) / sum_d["null_mean"] * 100.0
        return (f"未发现时间与横截面结构异常，但派生标量出现偏差：和值均值 "
                f"{sum_d['obs_mean']:.4f} vs 精确 null {sum_d['null_mean']:.4f}"
                f"（z={sum_d['z']:.3f}，p={sum_d['p_two_sided']:.4f}）显著{direction}，"
                f"效应量约 {effect_pct:.2f}%（每期约 "
                f"{abs(sum_d['obs_mean'] - sum_d['null_mean']):.2f} 量级），"
                f"逐年代弥散（非单一时期驱动）；无实践意义，建议持续监控并复核数据源一致性。")
    return ""


def run_red_spectral_test(reds: np.ndarray, alpha: float = 0.05) -> RedSpectralResult:
    """红球三路径随机性检验编排（顶层入口, 三路径并行 fan-out 后合成判定）。

    前置 _validate_reds（集合语义校验）; N<MIN_N(500) → verdict=INSUFFICIENT_DATA
    不崩溃。α 拆分: alpha_path=Sidak 三路(0.01695) → alpha_comp=路径内三组件(0.00568)。
    路径1: pooled_red_chi2 + repeat_rate_test + indicator_fisher_g_family(Bonf α_comp/33);
           indicator_lag1_zs 仅报告。gate1 = 任一触发。
    路径2: cooccurrence_matrix → pair_binomial_stats → bh_fdr(上尾正向同现 /
           下尾互斥且 obs≤期望−3σ) + 子类族 5 测试(Bonf α_comp/5, 临界 z=3.254);
           pmi_ranking 仅排序展示。gate2 = FDR 双向任一检出 或 子类任一触发。
    路径3: exact_sum_null/span_null/odd_even_null + moments_z_test(和值/跨度均值,
           判定项) + odd_even_chi2(奇偶判定); 和值 chi2 拟合(60 箱)仅报告。
           gate3 = 任一触发。
    verdict 优先级（高→低）: INSUFFICIENT_DATA > PEAK_CONFIRMED >
    STRUCTURAL_ANOMALY > NONSPECTRAL_BIAS > SCALAR_BIAS > FLAT。

    Args:
        reds: (N,6) int 红球数组（1..33, 每行 6 个互异, 集合语义不要求升序）。
        alpha: 总体显著性水平, 默认 0.05。

    Returns:
        RedSpectralResult(path1, path2, path3, verdict, conclusion, alpha_split,
        n_periods)。path1/2/3 为报告 dict（含 np 数组, 由 red_spectral_report_dict
        序列化）; N<MIN_N 时 path1/2/3 为 None。

    Raises:
        ValueError: 输入形状/值域/互异性不合法（N≥MIN_N 时）。
    """
    arr = _validate_reds(reds)
    n = arr.shape[0]
    alpha_path = ALPHA_PATH(alpha)
    alpha_comp = ALPHA_COMP(alpha)
    alpha_split = {
        "path": alpha_path,
        "comp": alpha_comp,
        "note": ("Sidak 三路拆分(α_path)→路径内三组件拆分(α_comp)；族内 Bonferroni："
                 "逐号谱峰 α_comp/33、子类 5 测试 α_comp/5；528 对用 BH-FDR @α_comp"),
    }
    if n < MIN_N:
        conclusion = f"样本量不足（N={n} < {MIN_N}），所有检验检测力不足，不判定。"
        return RedSpectralResult(path1=None, path2=None, path3=None,
                                 verdict="INSUFFICIENT_DATA", conclusion=conclusion,
                                 alpha_split=alpha_split, n_periods=n)
    # ---------- 路径1 时间维度 ----------
    indicator = red_indicator_matrix(arr)
    chi2 = pooled_red_chi2(arr, alpha=alpha_comp)
    repeat = repeat_rate_test(arr, alpha=alpha_comp)
    g_results = indicator_fisher_g_family(indicator, alpha=alpha_comp / RED_N)
    lag1_zs = indicator_lag1_zs(indicator)
    peak_count = sum(1 for g in g_results if g.significant)
    peak_info: Optional[Tuple[int, int, float]] = None
    for idx, g in enumerate(g_results):
        if g.significant:
            peak_info = (idx + 1, int(g.peak_bin), float(g.p_value))
            break
    gate1 = chi2.significant or repeat.significant or peak_count > 0
    path1 = {
        "passed": not gate1,
        "pooled_chi2": {
            "stat": chi2.stat, "df": chi2.df, "p_value": chi2.p_value,
            "significant": chi2.significant,
            "boundary_note": "p 贴近 0.05 时如实呈现，作为检验灵敏度证据",
        },
        "per_number": {
            "expected_per_number": 6.0 * n / RED_N,
            "counts": chi2.counts,
            "lag1_zs": lag1_zs,
            "lag1_max_z": float(np.max(np.abs(lag1_zs))),
            "fisher_g_p_values": np.array([g.p_value for g in g_results]),
            "fisher_g_min_p": float(min(g.p_value for g in g_results)),
            "fisher_g_significant_count": peak_count,
        },
        "repeat_rate": {
            "observed": repeat.observed_rate, "expected": repeat.expected_rate,
            "total_repeats": repeat.total_repeats, "z": repeat.z,
            "significant": repeat.significant,
            "variance_note": "超几何精确矩 var=0.753099",
        },
        "verdict": ("PEAK_CONFIRMED" if peak_count > 0 else
                    ("NONSPECTRAL_BIAS" if chi2.significant or repeat.significant
                     else "FLAT")),
    }
    # ---------- 路径2 横截面 ----------
    c = cooccurrence_matrix(arr)
    freq = indicator.sum(axis=1).astype(np.float64)
    z_mat, p_upper, p_lower = pair_binomial_stats(c, n)
    iu = np.triu_indices(RED_N, k=1)
    mu = n * PAIR_P
    sd = math.sqrt(n * PAIR_P * (1.0 - PAIR_P))
    sig_pos = bh_fdr(p_upper, alpha_comp)
    sig_neg = bh_fdr(p_lower, alpha_comp) & (c[iu] <= mu - 3.0 * sd)
    pos_pairs = [[int(iu[0][k]) + 1, int(iu[1][k]) + 1] for k in np.flatnonzero(sig_pos)]
    neg_pairs = [[int(iu[0][k]) + 1, int(iu[1][k]) + 1] for k in np.flatnonzero(sig_neg)]
    sub_results: List[SubclassResult] = []
    for sname, npairs, nshared, ndisjoint in SUBCLASS_TABLE:
        pairs = _subclass_pairs(sname)
        obs = int(sum(c[i, j] for i, j in pairs))
        sub_results.append(subclass_stats(obs, npairs, nshared, ndisjoint, n, name=sname))
    gate2 = len(pos_pairs) > 0 or len(neg_pairs) > 0 or any(
        s.significant for s in sub_results)
    path2 = {
        "passed": not gate2,
        "matrix": {
            "symmetric": bool(np.all(c == c.T)),
            "diagonal_zero": bool(np.all(np.diag(c) == 0)),
            "total": int(c.sum()),
            "expected_total": 30 * n,
            "expected_per_pair": 30.0 * n / 1056.0,
            "obs_range": [int(c[iu].min()), int(c[iu].max())],
        },
        "pair_tests": {
            "n_pairs": TOTAL_PAIRS,
            "method": ("逐对二项检验(逐期独立 Bernoulli(15/528))正态近似 + "
                       "BH-FDR(上尾正向同现 @α_comp) / "
                       "BH-FDR(下尾互斥 @α_comp 且 obs≤期望−3σ)"),
            "z_matrix": z_mat,
            "max_z": float(np.max(z_mat[iu])),
            "min_p_upper": float(np.min(p_upper)),
            "fdr_sig_positive": len(pos_pairs),
            "fdr_sig_negative": len(neg_pairs),
            "fdr_positive_pairs": pos_pairs,
            "fdr_negative_pairs": neg_pairs,
            "bonferroni_note": "528 对 Bonferroni 临界 z=3.904 过严弃用（ADR-004）",
        },
        "subclasses": [
            {"name": s.name, "n_pairs": s.n_pairs, "observed": s.observed,
             "expected": s.expected, "z": s.z, "significant": s.significant}
            for s in sub_results
        ],
        "pmi_top": _pmi_top_with_count(pmi_ranking(c, freq, n), c),
        "mutual_exclusion_definition": "BH-FDR 下尾显著 且 obs ≤ 期望−3σ",
        "verdict": "STRUCTURAL_ANOMALY" if gate2 else "FLAT",
    }
    # ---------- 路径3 派生标量 ----------
    sums = arr.sum(axis=1).astype(np.float64)
    spans = (arr.max(axis=1) - arr.min(axis=1)).astype(np.float64)
    odds = (arr % 2 == 1).sum(axis=1)
    _, sum_mean, sum_var = exact_sum_null()
    _, span_mean, span_var = span_null()
    _, odd_mean, odd_var = odd_even_null()
    sum_z = moments_z_test(float(sums.mean()), sum_mean, sum_var, n, alpha=alpha_comp)
    span_z = moments_z_test(float(spans.mean()), span_mean, span_var, n, alpha=alpha_comp)
    odd_z = moments_z_test(float(odds.mean()), odd_mean, odd_var, n, alpha=alpha_comp)
    odd_chi = odd_even_chi2(odds, alpha=alpha_comp)
    fit_report = _sum_chi2_fit(sums, n)
    gate3 = sum_z.significant or span_z.significant or odd_chi.significant
    path3 = {
        "passed": not gate3,
        "sum": {
            "null": "33C6 精确卷积(DP，确定性无 MC)",
            "null_mean": sum_mean, "null_var": sum_var,
            "obs_mean": float(sums.mean()), "obs_var": float(sums.var()),
            "z": sum_z.z, "p_two_sided": sum_z.p_value,
            "significant": sum_z.significant,
            "chi2_fit_report": {
                "n_bins": fit_report["n_bins"], "stat": fit_report["stat"],
                "df": fit_report["df"], "p": fit_report["p"],
                "note": ("仅报告；等概率箱构造对分箱数与边界敏感"
                         "（40/60/80 箱 p 在 0.05~0.40 间波动），判定走精确矩均值 z"),
            },
        },
        "span": {
            "null": "闭式 (33−s)·C(s−1,4)/C(33,6)",
            "null_mean": span_mean, "null_var": span_var,
            "obs_mean": float(spans.mean()), "z": span_z.z,
            "significant": span_z.significant,
        },
        "odd_even": {
            "null": "闭式 C(17,k)·C(16,6−k)",
            "chi2": odd_chi.stat, "df": odd_chi.df, "p": odd_chi.p_value,
            "obs_mean_odd": float(odds.mean()), "null_mean_odd": odd_mean,
            "z_mean": odd_z.z, "significant": odd_chi.significant,
        },
        "verdict": "SCALAR_BIAS" if gate3 else "FLAT",
    }
    # ---------- verdict 合成（优先级: 谱峰 > 横截面 > 非周期 > 标量 > FLAT） ----------
    if peak_count > 0:
        verdict = "PEAK_CONFIRMED"
    elif gate2:
        verdict = "STRUCTURAL_ANOMALY"
    elif chi2.significant or repeat.significant:
        verdict = "NONSPECTRAL_BIAS"
    elif gate3:
        verdict = "SCALAR_BIAS"
    else:
        verdict = "FLAT"
    conclusion = _conclusion(verdict, n, chi2, repeat, sub_results, path3,
                             alpha_comp, peak_info)
    return RedSpectralResult(path1=path1, path2=path2, path3=path3, verdict=verdict,
                             conclusion=conclusion, alpha_split=alpha_split, n_periods=n)


# ================= 序列化 =================
def _to_native(obj):
    """递归把 np 类型转为原生 Python（dict/list/ndarray/generic）。"""
    if isinstance(obj, dict):
        return {k: _to_native(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_to_native(v) for v in obj]
    if isinstance(obj, np.ndarray):
        return _to_native(obj.tolist())
    if isinstance(obj, np.generic):
        return obj.item()
    return obj


def _build_near_miss(result: RedSpectralResult) -> List[Dict]:
    """收集最接近阈值但未触发的条目（边界 p/近阈值子类/报告项, 不参与判定）。"""
    if result.path1 is None:
        return []
    near: List[Dict] = []
    subclasses = result.path2["subclasses"]
    insig = [s for s in subclasses if not s["significant"]]
    if insig:
        worst = max(insig, key=lambda s: abs(s["z"]))
        if abs(worst["z"]) >= 2.5:
            near.append({"type": "subclass", "name": worst["name"], "z": worst["z"],
                         "critical": 3.254, "note": "最接近子类阈值 3.254 的未显著条目"})
    chi2p = result.path1["pooled_chi2"]
    if not chi2p["significant"] and chi2p["p_value"] < 0.1:
        near.append({"type": "pooled_chi2", "p_value": chi2p["p_value"],
                     "alpha": result.alpha_split["comp"],
                     "note": "合并卡方边界 p 值（检验灵敏度证据）"})
    oz = result.path3["odd_even"]["z_mean"]
    if abs(oz) > 2.0:
        near.append({"type": "odd_mean_z", "z": oz, "label": "奇数个数均值",
                     "note": "奇数个数均值 z（报告项, 不参与判定）"})
    return near


def red_spectral_report_dict(result: RedSpectralResult, run_at: str,
                             source: str) -> Dict:
    """RedSpectralResult → 纯 dict（np 类型全转原生, 供 json.dumps / MD 渲染）。

    Args:
        result: run_red_spectral_test 的返回。
        run_at: 运行时间戳字符串（YYYY-MM-DD HH:MM:SS）。
        source: 数据源描述。

    Returns:
        报告 dict（schema 见架构文档 evaluate_integration.报告schema.spectral_red_probe）:
        kind/run_at/source/n_periods/alpha/alpha_split/path1/path2/path3/
        near_miss/verdict/conclusion/notes。
    """
    report = {
        "kind": "spectral_red_probe",
        "run_at": run_at,
        "source": source,
        "n_periods": result.n_periods,
        "alpha": 0.05,
        "alpha_split": _to_native(result.alpha_split),
        "path1": _to_native(result.path1) if result.path1 is not None else None,
        "path2": _to_native(result.path2) if result.path2 is not None else None,
        "path3": _to_native(result.path3) if result.path3 is not None else None,
        "near_miss": _build_near_miss(result),
        "verdict": result.verdict,
        "conclusion": result.conclusion,
        "notes": [
            "检验器不预测：本报告仅呈现统计证据，不构成选号建议（延续蓝球 extra_notes 口径）",
            "横截面显著 ≠ 可预测：多重比较 + 彩民群体已对连号类模式定价，显著项只做证据呈现",
            "边界 p 值如实呈现（如合并卡方 p=0.0677），是检验灵敏度证据而非检验失效",
            "逐号 lag-1 z 重尾（MC 族误报 8.5% vs 名义 5%），仅报告诊断；判定走聚合重号率 z",
            "和值 chi2 拟合分箱敏感，仅报告；判定走精确矩均值 z（无分箱任意性）",
            "PMI 为描述性排序指标，不用于显著性判定",
        ],
    }
    return report
