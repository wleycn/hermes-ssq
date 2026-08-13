#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""蓝球频谱平坦性三关检验纯函数库（SSQ 随机性检验探针）。

定位: 把「摇奖机公平」变成统计证据 —— 非预测器。无状态、无 IO、纯 numpy 计算;
只依赖 numpy / scipy / math / decimal（无 pandas、无数据 IO、无 evaluate 依赖）。

统计口径（架构文档 arch_spectral_probe.json, ADR 已拍板）:
- ADR-001 主编码 = 复平面单位圆 z = exp(2πi·x/16), x ∈ 1..16; one-hot 仅作门3 交叉复核。
- ADR-002 Fisher's g 精确 p 在全段原始周期图（m=N-1）上计算; Welch 平均谱仅作峰位交叉核对。
- ADR-003 门1 自相关用复圆自相关 |R(τ)|·√N（SE=1/√N 精确成立, 实测校准 0/1000 误报）。
α 控制: Sidak 双门拆分（α_gate≈0.02532）→ 门1 双子检验再 Sidak（α_sub≈0.01274）
→ 自相关族内 Bonferroni → 门3 条件检验 α=0.05（组合误报实测 4/100≈5%）。
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from decimal import Decimal, getcontext
from typing import Dict, List, Optional, Tuple

import numpy as np
import scipy.signal
import scipy.stats

# ================= 常量与 α 拆分 =================
BLUE_N = 16   # 蓝球号码数 1..16（分类标签）
MIN_N = 500   # 样本量硬边界: 低于则返回 INSUFFICIENT_DATA 不判定


def GATE_ALPHA(alpha: float = 0.05) -> float:
    """Sidak 双门拆分: α_gate = 1 - sqrt(1-α)。α=0.05 → 0.02532。

    Args:
        alpha: 显著性水平, 默认 0.05。

    Returns:
        Sidak 拆分后的门限 α_gate。
    """
    return 1.0 - math.sqrt(max(0.0, 1.0 - alpha))


def SUB_ALPHA(alpha: float = 0.05) -> float:
    """门1 双子检验再 Sidak: α_sub = 1 - sqrt(1-α_gate)。α=0.05 → 0.01274。

    Args:
        alpha: 显著性水平, 默认 0.05。

    Returns:
        Sidak 二次拆分后的门限 α_sub。
    """
    return 1.0 - math.sqrt(max(0.0, 1.0 - GATE_ALPHA(alpha)))


# ================= 结果 dataclass =================
@dataclass(frozen=True)
class Chi2Result:
    """蓝球 1..n 频率均匀性卡方检验结果。"""
    stat: float
    df: int
    p_value: float
    counts: np.ndarray          # (n,) 每号出现次数
    expected: float             # 期望频次 N/n
    significant: bool


@dataclass(frozen=True)
class LagAutocorrResult:
    """复圆自相关（lag 1..max_lag, 族内 Bonferroni）。"""
    max_lag: int
    rhos: np.ndarray            # (max_lag,) 复自相关 R(τ)
    z_scores: np.ndarray        # (max_lag,) |R(τ)|·√N
    max_z: float
    max_z_lag: int
    alpha_family: float
    critical_z: float
    significant: bool


@dataclass(frozen=True)
class FisherGResult:
    """Fisher's g 谱峰检验结果。peak_bin 为原始 FFT bin（DC=0 坐标系）。"""
    g: float
    p_value: float
    m: int                      # 参与检验的非 DC bin 数
    peak_bin: Optional[int]
    peak_freq: Optional[float]  # peak_bin / N（cycles/sample）
    peak_phase_deg: Optional[float]
    implicated_number: Optional[int]  # 仅 complex_signal=True: 峰值相位反推号码 x*
    significant: bool


@dataclass(frozen=True)
class WelchResult:
    """Welch 平均周期图（交叉核对层, 不作判定）。"""
    window: int
    noverlap: int
    n_windows: int
    freqs: np.ndarray
    psd: np.ndarray
    peak_bin: int
    peak_freq: float
    peak_ratio_median: float    # P[peak]/median(P[1:])
    peak_fraction: float        # P[peak]/sum(P[1:])


@dataclass(frozen=True)
class InvarianceResult:
    """编码不变性矩阵: 旋转/反射理论不变（|Δg|≤1e-9 硬断言口径）;
    任意标签重排不承诺不变（记录漂移指纹, 不作断言, 由门3 兜底）。"""
    rotation: Dict              # {g_delta_max, p_stable, stable}
    reflection: Dict            # {g_delta, stable}
    permutation: Dict           # {g_shift, peak_bin_shift, stable}
    stable: bool                # rotation.stable and reflection.stable


@dataclass(frozen=True)
class ThreeGateResult:
    """三关检验结果聚合（唯一事实来源, spectral_report_dict 序列化）。"""
    n_periods: int
    alpha: float
    alpha_gate: float
    alpha_sub: float
    gate1: Optional[Dict]       # {passed, chi2: Chi2Result, autocorr: LagAutocorrResult}
    gate2: Optional[Dict]       # {passed, fisher_g, welch: List[WelchResult], peak_bin_agrees, unstable_peak}
    gate3: Optional[Dict]       # {implicated_number, g, m, p_value, peak_bin, confirmed, note}
    invariance: Optional[InvarianceResult]
    rolling: Optional[Dict]     # {window, step, n_windows, min_p, frac_below_gate_alpha, note}
    verdict: str                # FLAT | PEAK_CONFIRMED | PEAK_ARTIFACT | NONSPECTRAL_BIAS | INSUFFICIENT_DATA
    conclusion: str


# ================= 编码 =================
def _validate_blue_series(series: np.ndarray, n: int = BLUE_N) -> np.ndarray:
    """校验蓝球序列: 非空一维整数数组, 取值 ∈ 1..n。违规抛 ValueError。"""
    arr = np.asarray(series)
    if arr.ndim != 1 or arr.size == 0:
        raise ValueError(f"series 必须是非空一维数组, 实际 shape={arr.shape}")
    if not np.issubdtype(arr.dtype, np.integer):
        raise ValueError(f"series 必须为整数类型（蓝球是分类标签）, 实际 dtype={arr.dtype}")
    if int(arr.min()) < 1 or int(arr.max()) > n:
        raise ValueError(f"蓝球取值必须 ∈ 1..{n}, 实际范围 [{int(arr.min())}, {int(arr.max())}]")
    return arr


def circular_encode(series: np.ndarray, n: int = BLUE_N) -> np.ndarray:
    """复平面单位圆编码: z_t = exp(2πi·x_t/n), x_t ∈ 1..n。

    性质: |z|=1; x→x+c（模 n）等价整体乘常数相位 e^(2πi·c/n), 功率谱不变
    （编码不变性测试基础）; H0 下 E[z]=0, 复圆自相关零均值、CLT 良好。
    Args:
        series: 号码序列(1..n 整数)。
                n: 类别数(蓝球 16)。

    Returns:
        复平面单位圆编码序列 z_t = exp(2πi·x_t/n), 长度与 series 相同。

    """
    arr = _validate_blue_series(series, n)
    return np.exp(2j * np.pi * arr / n)


def one_hot_series(series: np.ndarray, n: int = BLUE_N) -> np.ndarray:
    """one-hot 指示矩阵: (n, N) float64 0/1, 行 j 为号码 j+1 的指示序列。

    性质: 列和为 1; 对标签重排 π 天然不变: one_hot(π(x)) 的行重排 == π·one_hot(x)。
    Args:
        series: 号码序列(1..n 整数)。
                n: 类别数(蓝球 16)。

    Returns:
        (n, N) float64 0/1 矩阵, 行 j 为号码 j+1 的指示序列。

    """
    arr = _validate_blue_series(series, n)
    N = arr.size
    oh = np.zeros((n, N), dtype=np.float64)
    oh[arr - 1, np.arange(N)] = 1.0
    return oh


def periodogram(series: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """原始周期图: X = fft(series); P = |X|²。返回 (P, X)。

    DC bin=0 由调用方排除（Fisher g 与 Welch 峰位均从 bin 1 起算）。
    Args:
        series: 输入序列(实数或复数)。

    Returns:
        (P, X) 元组: P 为幅度谱 |X|², X 为 FFT 结果。

    """
    arr = np.asarray(series)
    if arr.ndim != 1 or arr.size < 2:
        raise ValueError(f"series 长度必须 ≥ 2, 实际 {arr.size}")
    X = np.fft.fft(arr)
    P = np.abs(X) ** 2
    return P, X


# ================= Fisher's g =================
def fisher_g_pvalue(g: float, m: int, prec: int = 60) -> float:
    """Fisher's g 精确 p = Σ_{k=1}^{⌊1/g⌋} (-1)^(k-1)·C(m,k)·(1-k·g)^(m-1)。

    decimal 精确累加: float 下 C(3487,k) 溢出（e^900 量级）, 故 math.comb 取精确
    大整数转 Decimal、(1-kg) 用 Decimal 幂, 避免交替级数的灾难性抵消。
    边界: g≤0 或 m<2 → 1.0; g≥1（kmax=0）→ 1.0。
    实测锚点: m=10, g=0.25 → p≈0.663414（闭式手算）;
    m=3487 MC 0.95 分位 g=0.003245 → p≈0.0411（轻微保守, 方向安全）。
    Args:
        g: Fisher g 统计量(>0)。
                m: 谱 bin 数。
                prec: Decimal 精度位数, 默认 50。

    Returns:
        Fisher's g 精确 p 值(0..1)。

    """
    if g <= 0.0 or m < 2:
        return 1.0
    kmax = int(math.floor(1.0 / g))
    if kmax < 1:
        return 1.0
    ctx = getcontext()
    old_prec = ctx.prec
    ctx.prec = prec
    try:
        gd = Decimal(str(g))  # str 保证精确十进制表示
        total = Decimal(0)
        expo = m - 1
        for k in range(1, kmax + 1):
            base = Decimal(1) - Decimal(k) * gd
            if base <= 0:
                break  # kg≥1 后 (1-kg)^(m-1)=0, 其后各项均为 0
            term = Decimal(math.comb(m, k)) * (base ** expo)
            total += term if (k & 1) else -term
        return float(total)
    finally:
        ctx.prec = old_prec


def fisher_g_test(series: np.ndarray, alpha: float = 0.05,
                  complex_signal: bool = True) -> FisherGResult:
    """Fisher's g 谱峰检验。

    complex_signal=True: 复序列, 用 bins 1..N-1, m=N-1（主检路径, 全段原始周期图）。
    False（one-hot 实序列）: 用 bins 1..(N-1)//2（不含 Nyquist）, m=(N-1)//2。
    g = max(P_nz)/sum(P_nz); p = fisher_g_pvalue(g, m); significant = p < alpha。
    complex_signal=True 时由峰值相位反推号码（门3 复核锚）:
      x* = round(16·angle(X[peak_bin])/(2π)) mod 16, 0 映射为 16。
    边界: 全零功率（sum≤0）→ g=0, p=1.0, significant=False。
    Args:
        series: 输入序列。
                alpha: 显著性水平。
                complex_signal: True 用全段 bins 1..N-1(复序列主检); False 用实序列 bins 1..(N-1)//2。

    Returns:
        FisherGResult(g, p_value, m, peak_bin, significant, phase_angle, phase_number)。

    """
    arr = np.asarray(series, dtype=np.complex128)
    if arr.ndim != 1 or arr.size < 2:
        raise ValueError(f"series 长度必须 ≥ 2, 实际 {arr.size}")
    N = arr.size
    P, X = periodogram(arr)
    P_nz = P[1:] if complex_signal else P[1:(N + 1) // 2]
    total = float(P_nz.sum())
    if total <= 0:
        return FisherGResult(g=0.0, p_value=1.0, m=len(P_nz), peak_bin=None,
                             peak_freq=None, peak_phase_deg=None,
                             implicated_number=None, significant=False)
    g = float(P_nz.max() / total)
    m = len(P_nz)
    peak_bin = int(np.argmax(P_nz)) + 1  # 原始 FFT bin（DC=0 坐标系）
    p_value = fisher_g_pvalue(g, m)
    peak_freq = float(peak_bin / N)
    peak_phase_deg = float(np.degrees(np.angle(X[peak_bin])))
    implicated_number: Optional[int] = None
    if complex_signal:
        x_star = int(round(16.0 * np.angle(X[peak_bin]) / (2.0 * np.pi))) % 16
        implicated_number = 16 if x_star == 0 else x_star
    return FisherGResult(g=g, p_value=p_value, m=m, peak_bin=peak_bin,
                         peak_freq=peak_freq, peak_phase_deg=peak_phase_deg,
                         implicated_number=implicated_number,
                         significant=bool(p_value < alpha))


def welch_spectrum(series: np.ndarray, windows: Tuple[int, ...] = (64, 128),
                   overlap: float = 0.5) -> List[WelchResult]:
    """Welch 平均周期图（交叉核对层, 不作判定）。

    复数输入无共轭对称 → scipy 返回全部 W 个频点（DC 外 W-1 个有效）。
    n_windows = 1 + (N-W)//(W·(1-overlap)); N<W 时 nperseg=N、noverlap=0 兜底。
    峰值: peak_bin = argmax(P[1:])+1; ratio/median 与 fraction 供解读。
    实测锚点: N=3488, W=64 → 108 窗; W=128 → 53 窗。
    Args:
        series: 输入序列。
                windows: 窗口长度元组, 如 (64, 128, 256)。
                overlap: 相邻窗口重叠率(0..1)。

    Returns:
        每个窗口长度的 WelchResult 列表。

    """
    arr = np.asarray(series)
    if arr.ndim != 1 or arr.size < 2:
        raise ValueError(f"series 长度必须 ≥ 2, 实际 {arr.size}")
    results: List[WelchResult] = []
    for W in windows:
        W = int(W)
        nperseg = min(W, arr.size)
        noverlap = int(nperseg * overlap) if arr.size >= W else 0
        freqs, psd = scipy.signal.welch(arr, nperseg=nperseg, noverlap=noverlap)
        step = int(W * (1.0 - overlap))
        n_windows = 1 + (arr.size - W) // step if arr.size >= W and step > 0 else 1
        P = np.asarray(psd, dtype=np.float64)
        if P.size < 2:
            results.append(WelchResult(W, noverlap, n_windows, freqs, P,
                                       0, 0.0, 0.0, 0.0))
            continue
        peak_bin = int(np.argmax(P[1:])) + 1
        peak_freq = float(freqs[peak_bin])
        if peak_freq < 0:  # 复数双边谱频率 ∈ [-0.5,0.5), 归一化为正频表示(锚点 0.6406)
            peak_freq += 1.0
        med = float(np.median(P[1:]))
        peak_ratio_median = float(P[peak_bin] / med) if med > 0 else float("nan")
        denom = float(P[1:].sum())
        peak_fraction = float(P[peak_bin] / denom) if denom > 0 else 0.0
        results.append(WelchResult(W, noverlap, n_windows, freqs, P, peak_bin,
                                   peak_freq, peak_ratio_median, peak_fraction))
    return results


# ================= 门1 子检验 =================
def chi2_uniform_test(series: np.ndarray, n: int = BLUE_N,
                      alpha: float = 0.05) -> Chi2Result:
    """蓝球 1..n 频率均匀性卡方检验（df=n-1）。

    stat = Σ(counts-E)²/E, E=N/n; p = scipy.stats.chisquare(counts, f_exp)。

    Args:
        series: 号码序列(1..n)。
        n: 类别数。
        alpha: 显著性水平。

    Returns:
        Chi2Result(chi2, df, p_value, significant, counts)。
    """
    arr = _validate_blue_series(series, n)
    N = arr.size
    counts = np.bincount(arr, minlength=n + 1)[1:n + 1].astype(np.float64)
    expected = N / n
    stat = float(np.sum((counts - expected) ** 2 / expected))
    p_value = float(scipy.stats.chisquare(counts, f_exp=[expected] * n).pvalue)
    return Chi2Result(stat=stat, df=n - 1, p_value=p_value, counts=counts,
                      expected=float(expected), significant=bool(p_value < alpha))


def lag_autocorrelation(series: np.ndarray, max_lag: int = 20,
                        alpha: float = 0.05) -> LagAutocorrResult:
    """复圆自相关: R(τ) = Σ_{t=τ}^{N-1} z_t·conj(z_{t-τ})/(N-τ); z_score = |R(τ)|·√N。

    H0 下复圆自相关 SE=1/√N 精确成立（有界复变量 CLT, 实测 1000 组 99.9 分位
    3.11 < 临界 3.42, 误报 0/1000）。族内 Bonferroni:
    critical_z = norm.ppf(1 - α/(2·max_lag))。N≤max_lag 时仅算到 N-1。
    Args:
        series: 号码序列(1..n)。
                max_lag: 最大滞后阶数。
                alpha: 显著性水平。

    Returns:
        LagAutocorrResult(ac, z_scores, se, critical_z, max_z, significant, peak_lag)。

    """
    z = circular_encode(series)
    N = z.size
    lags = min(max_lag, N - 1)
    rhos = np.empty(lags, dtype=np.complex128)
    z_scores = np.empty(lags, dtype=np.float64)
    for i, tau in enumerate(range(1, lags + 1)):
        r = np.vdot(z[tau:], z[:-tau]) / (N - tau)
        rhos[i] = r
        z_scores[i] = abs(r) * math.sqrt(N)
    critical_z = float(scipy.stats.norm.ppf(1.0 - alpha / (2.0 * max_lag)))
    if lags == 0:
        return LagAutocorrResult(max_lag=max_lag, rhos=rhos, z_scores=z_scores,
                                 max_z=0.0, max_z_lag=0, alpha_family=alpha,
                                 critical_z=critical_z, significant=False)
    max_z = float(z_scores.max())
    max_z_lag = int(np.argmax(z_scores)) + 1
    return LagAutocorrResult(max_lag=max_lag, rhos=rhos, z_scores=z_scores,
                             max_z=max_z, max_z_lag=max_z_lag, alpha_family=alpha,
                             critical_z=critical_z,
                             significant=bool(max_z > critical_z))


# ================= 编码不变性 =================
def encoding_invariance(series: np.ndarray, seed: int = 0,
                        n: int = BLUE_N) -> InvarianceResult:
    """编码不变性矩阵。

    旋转（x→x+c 模 n）只引入常数相位、反射（x→n+1-x）只镜像幅度谱 → 功率谱
    理论不变（|Δg|≤1e-9 硬断言口径, FFT 误差 1e-12 级）; 重排（任意标签置换）
    不承诺不变, 记录 g_shift/peak_bin_shift 漂移指纹（漂移即编码敏感信号,
    由门3 one-hot 复核兜底）。
    Args:
        series: 号码序列(1..n)。
                seed: 重排随机种子。
                n: 类别数。

    Returns:
        InvarianceResult(rotation, reflection, permutation)。

    """
    arr = _validate_blue_series(series, n)
    fg0 = fisher_g_test(circular_encode(arr))
    g0, p0, kb0 = fg0.g, fg0.p_value, fg0.peak_bin

    g_deltas: List[float] = []
    p_deltas: List[float] = []
    for c in (1, 3, 8):
        rot = ((arr - 1 + c) % n) + 1
        fg = fisher_g_test(circular_encode(rot))
        g_deltas.append(abs(fg.g - g0))
        p_deltas.append(abs(fg.p_value - p0))
    g_delta_max = float(max(g_deltas))
    p_stable = bool(all(d <= 1e-9 for d in p_deltas))
    rotation = {"g_delta_max": g_delta_max, "p_stable": p_stable,
                "stable": bool(g_delta_max <= 1e-9)}

    refl = (n + 1 - arr).astype(arr.dtype)
    fg_ref = fisher_g_test(circular_encode(refl))
    g_delta_ref = float(abs(fg_ref.g - g0))
    reflection = {"g_delta": g_delta_ref, "stable": bool(g_delta_ref <= 1e-9)}

    rng = np.random.default_rng(seed)
    pi = rng.permutation(n) + 1  # 1-based 置换
    fg_perm = fisher_g_test(circular_encode(pi[arr - 1]))
    g_shift = float(abs(fg_perm.g - g0))
    peak_bin_shift = (int(fg_perm.peak_bin - kb0)
                      if fg_perm.peak_bin is not None and kb0 is not None else None)
    permutation = {"g_shift": g_shift, "peak_bin_shift": peak_bin_shift,
                   "stable": bool(g_shift <= 1e-9)}

    return InvarianceResult(rotation=rotation, reflection=reflection,
                            permutation=permutation,
                            stable=bool(rotation["stable"] and reflection["stable"]))


# ================= 三关编排 =================
def _insufficient_result(arr: np.ndarray, alpha: float, window: int) -> ThreeGateResult:
    """样本量不足报告: 不崩溃、不判定, 结论模板注明检测力不足。"""
    N = arr.size
    return ThreeGateResult(
        n_periods=N, alpha=alpha, alpha_gate=GATE_ALPHA(alpha),
        alpha_sub=SUB_ALPHA(alpha), gate1=None, gate2=None, gate3=None,
        invariance=None, rolling=None, verdict="INSUFFICIENT_DATA",
        conclusion=(f"样本量不足（N={N} < MIN_N={MIN_N} 或 N < 2·window={2 * window}），"
                    f"所有检验检测力不足, 不作判定。"),
    )


def _conclusion_text(verdict: str, N: int, chi2: Chi2Result, ac: LagAutocorrResult,
                     fg: FisherGResult, gate3: Dict, gate2: Dict,
                     max_lag: int) -> str:
    """verdict 对应中文结论模板（架构文档 run_three_gate_test 规格）。"""
    if verdict == "FLAT":
        return (f"{N} 期蓝球序列未发现显著非随机结构：卡方均匀 p={chi2.p_value:.4f}、"
                f"lag-1~{max_lag} 复圆自相关 max|z|={ac.max_z:.3f}（临界 {ac.critical_z:.3f}）、"
                f"复谱 Fisher's g p={fg.p_value:.4f}、Welch 峰位稳定={gate2['peak_bin_agrees']}。"
                f"统计证据支持摇奖机公平假设。")
    if verdict == "PEAK_CONFIRMED":
        return (f"在 bin {fg.peak_bin}（周期 {N / fg.peak_bin:.1f} 期）发现显著谱峰，"
                f"one-hot 复核号码 {gate3['implicated_number']} 复现，疑似物理偏差，"
                f"建议深挖（球重/磨损/数据源）并人工复核。")
    if verdict == "PEAK_ARTIFACT":
        return (f"复谱峰未在 one-hot 谱复现（p={gate3['p_value']:.4f}），"
                f"判定为编码伪影而非真实周期结构。")
    if verdict == "NONSPECTRAL_BIAS":
        return (f"无谱峰但频率分布/自相关偏离均匀（卡方 p={chi2.p_value:.4f}, "
                f"max|z|={ac.max_z:.3f}），属非周期偏差，建议复核数据源与开奖号段。")
    return (f"样本量不足（N={N} < MIN_N={MIN_N}），所有检验检测力不足, 不作判定。")


def run_three_gate_test(series: np.ndarray, alpha: float = 0.05, window: int = 64,
                        overlap: float = 0.5, max_lag: int = 20,
                        rolling_step: int = 16) -> ThreeGateResult:
    """三关随机性检验编排（架构文档权威规格）。

    前置: 校验 1..16; N<MIN_N(500) 或 N<2·window → verdict=INSUFFICIENT_DATA。
    α 拆分: alpha_gate=Sidak 双门; alpha_sub=门1 双子检验再 Sidak。
    门1 无编码对照: 卡方均匀(α_sub) + 复圆自相关(α_sub, 族内 Bonferroni)。
    门2 复谱主检: fisher_g_test(circular_encode, α_gate); Welch(window,128) 峰位
      交叉核对 |k*/N - b/W| ≤ 1/W, 不一致记 unstable_peak=True。
    门3 one-hot 复核（仅门2 未过时触发）: x*=相位反推号码; one-hot 指示谱
      Fisher g(α=0.05, 实序列) 且峰 bin ∈ {k*, N-k*}±1 → confirmed。
    rolling 附加诊断（仅报告不决策）: 窗长 window、步进 rolling_step 的滑动窗
      Fisher g p 值数组。
    verdict 优先级: PEAK_CONFIRMED > PEAK_ARTIFACT > NONSPECTRAL_BIAS > FLAT
      （真周期信号门1 自相关在 lag=周期处必触发, 该顺序保证不被 NONSPECTRAL_BIAS 掩盖）。
    Args:
        series: 号码序列(1..n)。
                alpha: 总显著性水平。
                window: Welch 窗口长度。
                overlap: Welch 重叠率。
                max_lag: 自相关最大滞后。
                rolling_step: 滚动检验步长。

    Returns:
        ThreeGateResult, 含三关结果与 verdict(FLAT/PEAK_CONFIRMED/PEAK_ARTIFACT/NONSPECTRAL_BIAS/INSUFFICIENT_DATA)。

    """
    arr = _validate_blue_series(series)
    N = arr.size
    if alpha <= 0.0 or alpha >= 1.0:
        raise ValueError(f"alpha 必须 ∈ (0,1), 实际 {alpha}")
    if N < MIN_N or N < 2 * window:
        return _insufficient_result(arr, alpha, window)

    alpha_gate = GATE_ALPHA(alpha)
    alpha_sub = SUB_ALPHA(alpha)

    # ---- 门1 无编码对照 ----
    chi2 = chi2_uniform_test(arr, alpha=alpha_sub)
    ac = lag_autocorrelation(arr, max_lag=max_lag, alpha=alpha_sub)
    gate1_passed = bool((not chi2.significant) and (not ac.significant))
    gate1 = {"passed": gate1_passed, "chi2": chi2, "autocorr": ac}

    # ---- 门2 复谱主检 ----
    z = circular_encode(arr)
    fg = fisher_g_test(z, alpha=alpha_gate)
    welch_res = welch_spectrum(z, windows=(window, 128), overlap=overlap)
    w64 = welch_res[0]
    if fg.peak_bin is not None:
        agrees = abs(fg.peak_bin / N - w64.peak_bin / w64.window) <= 1.0 / w64.window
    else:
        agrees = False
    gate2_passed = bool(not fg.significant)
    gate2 = {"passed": gate2_passed, "fisher_g": fg, "welch": welch_res,
             "peak_bin_agrees": bool(agrees), "unstable_peak": bool(not agrees)}

    # ---- 门3 one-hot 复核（仅门2 未过时触发）----
    # 同频复现口径: one-hot 指示序列对周期结构存在多个等能量谐波 bin, 全局
    # argmax 会在谐波间漂移（实测混噪序列 argmax ∈ {218,436,...,1526} 随机）,
    # 故取 {k*, N-k*}±1 邻域窗口内最大功率的 Fisher 显著性作为 confirmed——
    # 语义 = 「对应号码指示谱在同频位置出现显著峰」（架构 20/20 检出锚点的口径）。
    if fg.significant and fg.implicated_number is not None and fg.peak_bin is not None:
        x_star = fg.implicated_number
        oh_series = one_hot_series(arr)[x_star - 1]
        P_oh, _ = periodogram(oh_series)
        P_nz = P_oh[1:(N + 1) // 2]
        total_oh = float(P_nz.sum())
        candidates = set()
        for kb in (fg.peak_bin, N - fg.peak_bin):
            for d in (-1, 0, 1):
                b = kb + d
                if 1 <= b < (N + 1) // 2:
                    candidates.add(b)
        if total_oh <= 0 or not candidates:
            oh_g, oh_p, oh_bin = 0.0, 1.0, None
        else:
            oh_bin = int(max(candidates, key=lambda b: float(P_oh[b])))
            oh_g = float(P_oh[oh_bin] / total_oh)
            oh_p = fisher_g_pvalue(oh_g, len(P_nz))
        confirmed = bool(oh_p < 0.05 and oh_bin is not None)
        gate3: Dict = {
            "implicated_number": x_star, "g": oh_g, "m": len(P_nz),
            "p_value": oh_p, "peak_bin": oh_bin,
            "confirmed": confirmed,
            "note": (f"one-hot 复核号码 {x_star}: g={oh_g:.6f}, p={oh_p:.6f}, "
                     f"峰 bin={oh_bin}（原始峰 {fg.peak_bin}）"),
        }
    else:
        gate3 = {"implicated_number": None, "g": None, "m": None, "p_value": None,
                 "peak_bin": None, "confirmed": False, "note": None}

    # ---- rolling 附加诊断（仅报告不决策, 非平稳可视化）----
    ps = [fisher_g_test(circular_encode(arr[s:s + window])).p_value
          for s in range(0, N - window + 1, rolling_step)]
    rolling: Dict = {
        "window": window, "step": rolling_step, "n_windows": len(ps),
        "min_p": float(min(ps)) if ps else None,
        "frac_below_gate_alpha": float(np.mean([p < alpha_gate for p in ps])) if ps else None,
        "note": f"仅报告不决策（{len(ps)} 窗下 min_p 受多重比较支配）",
    }

    # ---- 编码不变性 ----
    invariance = encoding_invariance(arr)

    # ---- verdict 判定（优先级见函数 docstring）----
    if fg.significant:
        verdict = "PEAK_CONFIRMED" if gate3["confirmed"] else "PEAK_ARTIFACT"
    elif not gate1_passed:
        verdict = "NONSPECTRAL_BIAS"
    else:
        verdict = "FLAT"

    conclusion = _conclusion_text(verdict, N, chi2, ac, fg, gate3, gate2, max_lag)

    return ThreeGateResult(
        n_periods=N, alpha=alpha, alpha_gate=alpha_gate, alpha_sub=alpha_sub,
        gate1=gate1, gate2=gate2, gate3=gate3, invariance=invariance,
        rolling=rolling, verdict=verdict, conclusion=conclusion,
    )


# ================= 序列化 =================
def _native(x):
    """np 数组 → list, np.generic → 标量, 非有限 float → None（JSON 安全）。"""
    if isinstance(x, np.ndarray):
        return x.tolist()
    if isinstance(x, np.generic):
        return x.item()
    if isinstance(x, dict):
        return {k: _native(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [_native(v) for v in x]
    if isinstance(x, float) and not math.isfinite(x):
        return None
    return x


def spectral_report_dict(result: ThreeGateResult, run_at: str, source: str) -> Dict:
    """ThreeGateResult → 纯 dict（报告 schema 见架构文档 spectral_probe）。

    np 数组转 list、np.generic 转 float/int; INSUFFICIENT_DATA 时 gate1/2/3、
    invariance、rolling 为 None。直接供 evaluate.py json.dumps / render_md_spectral。
    Args:
        result: ThreeGateResult。
                run_at: 运行时间戳字符串。
                source: 数据来源描述。

    Returns:
        可 JSON 序列化的报告 dict(schema 见架构文档 spectral_probe)。

    """
    report: Dict = {
        "kind": "spectral_probe",
        "run_at": run_at,
        "source": source,
        "n_periods": result.n_periods,
        "alpha": result.alpha,
        "alpha_split": {
            "gate": round(result.alpha_gate, 5),
            "sub": round(result.alpha_sub, 5),
            "note": "Sidak 双门拆分；门1 双子检验再 Sidak；自相关族内 Bonferroni",
        },
        "gate1": None,
        "gate2": None,
        "gate3": None,
        "invariance": None,
        "rolling": None,
        "verdict": result.verdict,
        "conclusion": result.conclusion,
        "notes": [
            "检验器不预测：本报告仅诊断蓝球序列随机性, 不构成下注建议",
            "主编码=复平面单位圆 z=e^(2πi·x/16)；one-hot 仅作门3 交叉复核",
            "Welch 平均谱仅作峰位交叉核对；Fisher's g 精确 p 在全段原始周期图上计算（ADR-002）",
        ],
    }
    if result.gate1 is None:
        return report

    c1, a1 = result.gate1["chi2"], result.gate1["autocorr"]
    report["gate1"] = {
        "passed": bool(result.gate1["passed"]),
        "chi2": {"stat": _native(c1.stat), "df": _native(c1.df),
                 "p_value": _native(c1.p_value), "significant": bool(c1.significant)},
        "autocorr": {"max_lag": _native(a1.max_lag), "max_z": _native(a1.max_z),
                     "max_z_lag": _native(a1.max_z_lag),
                     "critical_z": _native(a1.critical_z),
                     "significant": bool(a1.significant)},
    }

    fg2 = result.gate2["fisher_g"]
    report["gate2"] = {
        "passed": bool(result.gate2["passed"]),
        "fisher_g": {"g": _native(fg2.g), "m": _native(fg2.m),
                     "p_value": _native(fg2.p_value), "peak_bin": _native(fg2.peak_bin),
                     "peak_freq": _native(fg2.peak_freq),
                     "peak_phase_deg": _native(fg2.peak_phase_deg),
                     "significant": bool(fg2.significant)},
        "welch": [{"window": _native(w.window), "noverlap": _native(w.noverlap),
                   "n_windows": _native(w.n_windows), "peak_bin": _native(w.peak_bin),
                   "peak_freq": _native(w.peak_freq),
                   "peak_ratio_median": _native(w.peak_ratio_median),
                   "peak_fraction": _native(w.peak_fraction)}
                  for w in result.gate2["welch"]],
        "peak_bin_agrees": bool(result.gate2["peak_bin_agrees"]),
        "unstable_peak": bool(result.gate2["unstable_peak"]),
    }

    g3 = result.gate3
    report["gate3"] = {"implicated_number": _native(g3["implicated_number"]),
                       "g": _native(g3["g"]), "m": _native(g3["m"]),
                       "p_value": _native(g3["p_value"]),
                       "peak_bin": _native(g3["peak_bin"]),
                       "confirmed": bool(g3["confirmed"]), "note": g3["note"]}

    inv = result.invariance
    report["invariance"] = {
        "rotation": {"g_delta_max": _native(inv.rotation["g_delta_max"]),
                     "stable": bool(inv.rotation["stable"])},
        "reflection": {"g_delta": _native(inv.reflection["g_delta"]),
                       "stable": bool(inv.reflection["stable"])},
        "permutation": {"g_shift": _native(inv.permutation["g_shift"]),
                        "peak_bin_shift": _native(inv.permutation["peak_bin_shift"]),
                        "stable": bool(inv.permutation["stable"])},
    }
    report["rolling"] = {k: _native(v) for k, v in result.rolling.items()}
    return report
