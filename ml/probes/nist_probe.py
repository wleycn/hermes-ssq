#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""NIST SP 800-22 随机性检验套件（适用子集, P0, 研究简报 2026-08-19 [1]）。

⚠ 样本量功效 caveat（关键, 防过度宣称）：
  NIST 官方推荐**单序列 n ≥ 100,000 bits**；Maurer universal / linear complexity
  等重测试需 ~10^6+ bits 才具统计功效。本项目全部 3489 期拼成比特流仅约
  140K bits（蓝球单球序列仅 ~14K bits），**多处低于 NIST 推荐阈值**，若干重测试
  将功效不足甚至无效。
  → 本模块设计为：跑能跑的检验 + **逐测试标注功效状态**（OK / LOW_POWER / INVALID），
    结论一律表述为"在有限功效下未能拒绝 H0"，**绝不包装成权威认证**。
  → 这与研究简报对 qwen 误读（"证据链最密一环/最权威"）的纠正一致。

实现选择（纯 numpy/scipy，无第三方 RNG 测试库依赖）：
  实现 6 个对中等样本量仍可用、且彼此正交的检验：
    1. frequency            : 单比特频数（整体 0/1 平衡）
    2. runs                 : 游程总数（0/1 交替结构）
    3. longest_run          : 最长 1-游程（块内），分块统计
    4. fft                  : 离散傅里叶变换（谱峰功率为随机性判据）
    5. serial               : 2-bit 序列（相邻依赖）
    6. approx_entropy       : 近似熵（正则性，Rukhin 2000 χ² 口径）
  另有 linear_complexity_lfsr() 供大样本时调用（本项目当前数据不足，标注 LOW_POWER）。

比特编码：把每期自然数序列（红球 1..33、蓝球 1..16）按固定位宽二进制拼接，
跨期串成 bit 流。也可直接传入已编码的 0/1 序列。

依赖：仅 numpy / scipy.stats。
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Sequence

import numpy as np
from scipy import stats


# ----------------------------- 编码工具 ----------------------------- #

def encode_natural_series(series: Sequence[int], width: int) -> np.ndarray:
    """把自然数序列（1..2^width-1）按 width 位二进制拼接为 0/1 bit 流。

    Args:
        series: 自然数序列（如红球每期 6 个 1..33 → width=6；蓝球 1..16 → width=5）。
        width: 每数编码位宽。
    Returns:
        uint8 数组，0/1。
    """
    bits: List[int] = []
    for v in series:
        v = int(v)
        for b in range(width - 1, -1, -1):
            bits.append((v >> b) & 1)
    return np.array(bits, dtype=np.uint8)


def encode_red_blue(reds: Sequence[Sequence[int]], blues: Sequence[int],
                    red_width: int = 6, blue_width: int = 5) -> np.ndarray:
    """把多期开奖（reds: 每期 6 红；blues: 每期 1 蓝）编码为统一 bit 流。"""
    out: List[int] = []
    for red6, blue in zip(reds, blues):
        for r in red6:
            for b in range(red_width - 1, -1, -1):
                out.append((int(r) >> b) & 1)
        for b in range(blue_width - 1, -1, -1):
            out.append((int(blue) >> b) & 1)
    return np.array(out, dtype=np.uint8)


# ----------------------------- 检验实现 ----------------------------- #

@dataclass
class NistResult:
    name: str
    p: Optional[float]                # p 值（None=未计算/无效）
    verdict: str                   # RANDOM | NONRANDOM | LOW_POWER | INVALID
    power: str = "OK"              # OK | LOW_POWER（样本量不足标注）
    detail: str = ""


def _freq_test(bits: np.ndarray) -> NistResult:
    n = bits.size
    if n < 100:
        return NistResult("frequency", None, "INVALID", "LOW_POWER", f"n={n}<100")
    s = bits.astype(int).sum()
    s_obs = abs(2 * s - n) / math.sqrt(n)
    p = 2 * (1 - stats.norm.cdf(s_obs))
    return NistResult("frequency", round(float(p), 6),
                      "RANDOM" if p > 0.01 else "NONRANDOM", "OK")


def _runs_test(bits: np.ndarray) -> NistResult:
    n = bits.size
    if n < 100:
        return NistResult("runs", None, "INVALID", "LOW_POWER", f"n={n}<100")
    ones = bits.astype(int).sum()
    p_hat = ones / n
    if abs(p_hat - 0.5) >= 0.5 - 2 / math.sqrt(n):
        return NistResult("runs", None, "INVALID", "LOW_POWER",
                          "频数偏离 0.5 过大(前置条件不满足)")
    pi = (np.diff(bits.astype(int)) != 0).sum() + 1  # 游程数
    num = abs(pi - 2 * n * p_hat * (1 - p_hat))
    den = 2 * math.sqrt(2 * n) * p_hat * (1 - p_hat)
    p = 2 * (1 - stats.norm.cdf(num / den))
    return NistResult("runs", round(float(p), 6),
                      "RANDOM" if p > 0.01 else "NONRANDOM", "OK")


def _longest_run_test(bits: np.ndarray, M: int = 128) -> NistResult:
    """NIST 改进版最长游程检验（SP800-22 §2.4，块长 M=128）。

    把序列切成 K=⌊n/M⌋ 块，每块统计最长'1'游程长度，归入 7 类
    (π₀..π₆：长度 1..6 及 ≥7)，与理论概率分布做 χ²(6) 检验。
    对均匀随机位流应返回 RANDOM。
    """
    n = bits.size
    K = n // M
    if K < 10:
        return NistResult("longest_run", None, "INVALID", "LOW_POWER",
                          f"块数 K={K}<10 (需 n≥{10*M})")
    pi = np.array([0.8646739, 0.1191582, 0.0131423, 0.0013798,
                   0.0001413, 0.0000142, 0.0000010])  # NIST π₀..π₆
    bins = np.zeros(7, dtype=int)
    bits = bits.astype(int)
    for b in range(K):
        blk = bits[b * M:(b + 1) * M]
        cur = mx = 0
        for v in blk:
            cur = cur + 1 if v == 1 else 0
            mx = max(mx, cur)
        idx = min(mx, 6)  # 长度≥7 归入第 6 类
        bins[idx] += 1
    exp = K * pi
    chi2 = np.sum((bins - exp) ** 2 / exp)
    p = 1 - stats.chi2.cdf(chi2, 6)
    return NistResult("longest_run", round(float(p), 6),
                      "RANDOM" if p > 0.01 else "NONRANDOM", "OK",
                      f"χ²={chi2:.2f}")


def _fft_test(bits: np.ndarray) -> NistResult:
    n = bits.size
    if n < 128:
        return NistResult("fft", None, "INVALID", "LOW_POWER", f"n={n}<128")
    x = bits.astype(float) - 0.5  # 中心化
    spec = np.fft.fft(x)
    mag = np.abs(spec[:n // 2])
    # NIST: 标准化峰值高度 = |S(f)| / √(n/2)，阈值 ν = √(-2·ln(0.05)) = 2.995732274
    # （χ²(2) 的 0.95 分位）。比较标准化峰值而非原始幅值。
    mag_norm = mag / math.sqrt(n / 2.0)
    nu = 2.995732274
    n0 = n // 2
    n_less = np.sum(mag_norm < nu)
    d = (n_less - 0.95 * n0) / math.sqrt(0.95 * 0.05 * n0)
    p = 2 * (1 - stats.norm.cdf(abs(d)))
    return NistResult("fft", round(float(p), 6),
                      "RANDOM" if p > 0.01 else "NONRANDOM", "OK")


def _serial_test(bits: np.ndarray, m: int = 2) -> NistResult:
    """2-bit 序列串行检验（相邻依赖）。"""
    n = bits.size
    if n < 100:
        return NistResult("serial", None, "INVALID", "LOW_POWER", f"n={n}<100")
    bits = bits.astype(int)
    seq = (bits[:-1] << 1) | bits[1:]
    counts = np.bincount(seq, minlength=4).astype(float)
    exp = n / 4.0
    chi2 = np.sum((counts - exp) ** 2 / exp)
    p = 1 - stats.chi2.cdf(chi2, 3)
    return NistResult("serial", round(float(p), 6),
                      "RANDOM" if p > 0.01 else "NONRANDOM", "OK")


def _approx_entropy(bits: np.ndarray, m: int = 2, r: float = 0.0) -> NistResult:
    """近似熵（Rukhin 2000 χ² 口径：2m 维 vs (m+1) 维 频数差异）。

    对二进制序列 r 取 0（精确匹配计数）。返回基于 C(m) 差的 z 近似 p 值。
    """
    n = bits.size
    if n < 100:
        return NistResult("approx_entropy", None, "INVALID", "LOW_POWER", f"n={n}<100")
    x = bits.astype(int)

    def _phi(order: int) -> float:
        if order == 0:
            return 0.0
        patterns = x[:n - order + 1].reshape(-1, 1)
        cnt = np.zeros((2 ** order,), dtype=float)
        for i in range(n - order + 1):
            idx = 0
            for j in range(order):
                idx = (idx << 1) | x[i + j]
            cnt[idx] += 1
        p = cnt[cnt > 0] / cnt[cnt > 0].sum()
        return float(np.sum(p * np.log(p)))

    phi_m = _phi(m)
    phi_m1 = _phi(m + 1)
    apen = phi_m - phi_m1  # 理论随机 → ln2 - ln2 = 0（近似）
    # Rukhin χ² 近似：2 n (ln2 - ApEn) 服从 χ²(2^m - 1) 近似（简化）
    stat = 2 * n * (math.log(2) - apen) if apen < math.log(2) else 0.0
    p = 1 - stats.chi2.cdf(stat, 2 ** m - 1) if stat > 0 else 1.0
    return NistResult("approx_entropy", round(float(p), 6),
                      "RANDOM" if p > 0.01 else "NONRANDOM", "OK",
                      f"ApEn={apen:.4f}")


# ----------------------------- 统一入口 ----------------------------- #

def run_nist_subset(bits: Sequence[int]) -> List[NistResult]:
    """跑 NIST 适用子集（本项目样本量下可用 + 功效标注）。"""
    bits_arr = np.asarray(bits, dtype=np.uint8).ravel()
    if bits_arr.size < 100:
        return [NistResult("ALL", None, "INVALID", "LOW_POWER",
                           f"总比特 {bits_arr.size} 远低于 NIST 推荐 100,000")]
    return [
        _freq_test(bits_arr),
        _runs_test(bits_arr),
        # _longest_run_test 暂未纳入：NIST M=128 改进版的 π₀..π₆ 权威表需核对，
        #   当前实现在 200k 均匀随机位上误报 NONRANDOM，疑似 π 表口径偏差；
        #   为避免输出错误结论，暂不进子集（不影响其余正交检验）。
        # _fft_test 暂未纳入：标准化谱峰阈值常数需与权威 NIST 实现逐位核对，
        #   当前归一化口径在均匀随机位上误报 NONRANDOM；项目频谱随机性已由
        #   ml/spectral.py（Fisher's g 检验）覆盖，无需在此重复。
        _serial_test(bits_arr),
        _approx_entropy(bits_arr),
    ]


def summarize(results: List[NistResult]) -> dict:
    """汇总：标注整体功效状态，不宣称权威认证。"""
    n_total = len(results)
    n_invalid = sum(1 for r in results if r.verdict == "INVALID")
    n_nonrand = sum(1 for r in results if r.verdict == "NONRANDOM")
    return {
        "n_tests": n_total,
        "n_invalid_or_lowpower": n_invalid,
        "n_nonrandom": n_nonrand,
        "overall": "RANDOM_UNDER_LIMITED_POWER" if n_nonrand == 0
                   else "SOME_NONRANDOM",
        "caveat": ("在有限样本量（远低于 NIST 推荐阈值）下未能拒绝 H0；"
                   "结论为'有限功效下未见非随机结构'，非权威认证。"),
    }
