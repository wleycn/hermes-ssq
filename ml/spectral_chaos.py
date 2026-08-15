#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""混沌理论 & 相空间重构检验模块（第 4 把随机性检验器）。

定位
----
独立模块（不继承 BaseModel）, 不是预测器, 而是「双色球是否存在确定性混沌结构」
的统计检验器。流程: Takens 延迟嵌入重建相空间 → Rosenstein 算法估计最大 Lyapunov
指数 → FFT 相位随机化生成 surrogate 序列 → 对比原始指标与替代序列分布
（z-score / 双侧经验 p 值）→ 输出 verdict。

判定规则
--------
- p > 0.05          → RANDOM（与纯随机替代序列无显著差异 = 无确定性结构）
- p <= 0.05 且 z>2  → CHAOTIC/STRUCTURED（需谨慎解读）
- 其余              → RANDOM

双对照设计（关键）
--------
FFT 相位随机化 surrogate 保留功率谱但改变幅值分布（原始是离散均匀整数,
替代是连续近似高斯）, 会导致「分布形状伪影」误报。因此 run_chaos_test
对主指标同时跑两个对照:
1. FFT 相位随机化（相位结构）;
2. 同分布随机洗牌 shuffle（时间结构, 洗牌只破坏顺序、完整保留整数分布）。
最终 verdict 要求两个对照**都**显著才判 CHAOTIC/STRUCTURED——只要同分布
洗牌对照无差异, 即无时间结构证据, 判 RANDOM（差异归因于分布形状伪影）。

诚实预告
--------
纯白噪声的 Rosenstein 最大 Lyapunov 指数同样为正, 因此不能只看 λ 符号,
必须依赖 surrogate 相对检验。双色球是独立均匀随机过程, 预期全部列
p > 0.05 → RANDOM——这本身就是科学结论, 不是失败。

依赖: 仅 numpy + scipy.spatial（自实现 Rosenstein / SampEn / FFT surrogate,
不依赖 nolds）。中文注释, Google docstring, type annotation。
"""
from __future__ import annotations

from typing import Dict, Optional, Tuple

import numpy as np
from scipy import stats

try:  # scipy 新版已将 cKDTree 并入 KDTree 类
    from scipy.spatial import cKDTree as _KDTree
except ImportError:
    from scipy.spatial import KDTree as _KDTree

# ================= 常量 =================
DIM_MIN = 3                # FNN 嵌入维搜索下限
DIM_MAX = 7                # FNN 嵌入维搜索上限
FNN_THRESHOLD = 0.10       # 假近邻比例阈值: 低于该值认为维数足够
FNN_TOL = 15.0             # FNN 距离增长比阈值 R_tol
MIN_TSEP = 10              # Rosenstein 最近邻最小时间间隔（排除时间相关邻居）
LYAP_FIT_RANGE = (8, 24)   # 扩展率曲线线性拟合区间 [k_lo, k_hi]
MI_MAX_TAU = 20            # 互信息搜索最大时延
MI_BINS = 16               # 互信息二维直方图分箱数（每维）
SAMPEN_M = 2               # 样本熵模板维数
SAMPEN_R_FRAC = 0.2        # 样本熵容差系数（r = r_frac * std）
ALPHA = 0.05               # 显著性水平
Z_THRESHOLD = 2.0          # z-score 阈值


def takens_embedding(series: np.ndarray, dim: int, tau: int) -> np.ndarray:
    """Takens 延迟嵌入: 将一维时间序列重构为 (n_samples, dim) 相空间轨道。

    Args:
        series: 一维时间序列（float）。
        dim: 嵌入维数 m。
        tau: 延迟时间（滞后步数）。

    Returns:
        形状 (N - (dim-1)*tau, dim) 的嵌入矩阵, 第 i 行为
        [x_i, x_{i+tau}, ..., x_{i+(dim-1)*tau}]。
    """
    x = np.asarray(series, dtype=float).ravel()
    n = len(x)
    if n <= (dim - 1) * tau:
        raise ValueError(f"序列长度 {n} 不足以进行 dim={dim}, tau={tau} 的嵌入")
    # 行索引 + 列滞后偏移, 一步生成全部窗口
    idx = np.arange(n - (dim - 1) * tau)[:, None] + tau * np.arange(dim)[None, :]
    return x[idx]


def mutual_info_first_min(series: np.ndarray, max_tau: int = MI_MAX_TAU,
                          bins: int = MI_BINS) -> int:
    """用二维直方图估计互信息 I(tau), 返回第一局部极小对应的延迟 tau。

    对独立随机序列, I(tau) 在 tau>=1 处都接近 0, 第一极小通常为 1;
    若不存在局部极小, 回退到全局最小 tau（单调时即 1）。

    Args:
        series: 一维时间序列。
        max_tau: 最大搜索时延。
        bins: 分箱数（每维）。

    Returns:
        建议延迟 tau（>= 1）。
    """
    x = np.asarray(series, dtype=float).ravel()
    vals: list[float] = []
    for t in range(1, max_tau + 1):
        hist, _, _ = np.histogram2d(x[:-t], x[t:], bins=bins)
        total = hist.sum()
        if total == 0:
            vals.append(0.0)
            continue
        pij = hist / total
        pi = pij.sum(axis=1, keepdims=True)   # P(x_t) 边缘分布
        pj = pij.sum(axis=0, keepdims=True)   # P(x_{t+tau}) 边缘分布
        nz = pij > 0
        # I = Σ p(x,y)·ln[p(x,y)/(p(x)p(y))]
        vals.append(float(np.sum(pij[nz] * np.log(pij[nz] / (pi * pj)[nz]))))
    # 找第一局部极小: I(t) < I(t-1) 且 I(t) <= I(t+1)
    for t in range(1, len(vals) - 1):
        if vals[t] < vals[t - 1] and vals[t] <= vals[t + 1]:
            return t + 1
    # 回退: 全局最小
    return int(np.argmin(vals)) + 1


def fnn_ratio(emb_m: np.ndarray, emb_m1: np.ndarray, tol: float = FNN_TOL) -> float:
    """假近邻（False Nearest Neighbors）比例。

    对 m 维嵌入的每个点找最近邻（排除自身）, 在 m+1 维下检查距离增长:
    若增长比 > tol 且新增坐标差超过典型尺度, 记为假近邻——说明 m 维
    展开不充分, 需要更高维。

    Args:
        emb_m: m 维嵌入矩阵。
        emb_m1: m+1 维嵌入矩阵（与 emb_m 同一起点, 列数多 1）。
        tol: 距离增长比阈值。

    Returns:
        假近邻比例 [0, 1]。
    """
    n = len(emb_m)
    if n < 10:
        return 0.0
    dists, idxs = _KDTree(emb_m).query(emb_m, k=2)   # 第 0 个是自身
    d0 = dists[:, 1]
    j = idxs[:, 1]
    d1 = np.linalg.norm(emb_m1 - emb_m1[j], axis=1)  # m+1 维下与同一邻居的距离
    growth = d1 / np.maximum(d0, 1e-12)
    scale = float(np.std(emb_m1[:, -1]))             # 新增坐标的典型尺度
    fn = (growth > tol) & (d1 > 2.0 * scale) if scale > 0 else (growth > tol)
    return float(fn.mean())


def choose_embed_dim(series: np.ndarray, tau: int) -> int:
    """用 FNN 方法选择嵌入维: 返回第一个假近邻比例 < FNN_THRESHOLD 的 m ∈ [3, 7]。

    若 3..6 维都不收敛, 说明序列没有有限维确定性结构——这本身就是随机证据,
    此时回退到最小维 DIM_MIN（对随机序列最稳健）。

    Args:
        series: 一维时间序列。
        tau: 延迟时间。

    Returns:
        嵌入维数 m。
    """
    for m in range(DIM_MIN, DIM_MAX):  # 3,4,5,6 试到 7 之前
        emb_m = takens_embedding(series, m, tau)
        emb_m1 = takens_embedding(series, m + 1, tau)
        if len(emb_m1) < 10:
            return m
        if fnn_ratio(emb_m[:len(emb_m1)], emb_m1) < FNN_THRESHOLD:
            return m
    return DIM_MIN  # FNN 未收敛: 无有限维确定性结构


def max_lyapunov_rosenstein(series: np.ndarray, dim: int, tau: int,
                            dt: float = 1.0, min_tsep: int = MIN_TSEP,
                            fit_range: Tuple[int, int] = LYAP_FIT_RANGE) -> float:
    """Rosenstein 算法估计最大 Lyapunov 指数。

    步骤: 延迟嵌入 → 每个参考点找最近邻（排除时间间隔 < min_tsep 的
    时间相关邻居）→ 沿轨道同步演化, 计算平均对数距离增长 <ln d(k)> →
    在扩散平台期做线性拟合, 斜率即 λ_max。

    Args:
        series: 一维时间序列。
        dim: 嵌入维数。
        tau: 延迟时间。
        dt: 采样间隔（斜率换算用, 默认 1）。
        min_tsep: 最近邻最小时间间隔。
        fit_range: 扩展率曲线拟合区间 (k_lo, k_hi)。

    Returns:
        最大 Lyapunov 指数; 数据不足或退化时返回 NaN。
        注意: 纯白噪声也为正, 必须结合 surrogate test 解读。
    """
    x = np.asarray(series, dtype=float).ravel()
    x = x - x.mean()
    emb = takens_embedding(x, dim, tau)
    n = len(emb)
    if n < 30:
        return float('nan')
    k_max = min(30, (n - 1) // 2)
    # 对每个参考点取前 k_nn 近邻, 选第一个满足时间间隔要求的
    k_nn = min(8, n)
    dists, idxs = _KDTree(emb).query(emb, k=k_nn)
    if k_nn == 1:
        dists, idxs = dists[:, None], idxs[:, None]
    idx_ar = np.arange(n)
    d0 = np.full(n, np.nan)   # 最近邻初始距离
    j0 = np.full(n, -1)       # 最近邻索引
    for k in range(1, k_nn):  # 第 0 个是自身, 跳过
        cand, dcand = idxs[:, k], dists[:, k]
        ok = (np.abs(cand - idx_ar) >= min_tsep) & np.isnan(d0)
        d0[ok] = dcand[ok]
        j0[ok] = cand[ok]
    # 参考点与邻居都要留出演化余量, 避免越界
    valid = np.isfinite(d0) & (j0 >= 0) & (idx_ar < n - 1 - k_max) & (j0 < n - 1 - k_max)
    ref = np.where(valid)[0]
    if len(ref) < 10:
        return float('nan')
    j = j0[ref]
    d_start = np.maximum(d0[ref], 1e-12)
    # 平均对数距离增长曲线 <ln d(k)>
    div = np.full(k_max + 1, np.nan)
    for k in range(k_max + 1):
        dd = np.linalg.norm(emb[ref + k] - emb[j + k], axis=1)
        ln = np.log(np.maximum(dd, 1e-12) / d_start)
        ln = ln[np.isfinite(ln)]
        if ln.size:
            div[k] = ln.mean()
    # 平台期线性拟合: 斜率 = λ_max
    k_lo, k_hi = fit_range
    k_hi = min(k_hi, k_max)
    ks = np.arange(k_lo, k_hi + 1)
    ys = div[k_lo:k_hi + 1]
    ok = np.isfinite(ys)
    if ok.sum() < 3:
        return float('nan')
    slope, _ = np.polyfit(ks[ok], ys[ok], 1)
    return float(slope / dt)


def fft_surrogates(series: np.ndarray, n_surrogates: int = 100,
                   rng: Optional[np.random.Generator] = None) -> np.ndarray:
    """FFT 相位随机化生成替代序列（保留功率谱, 随机化相位）。

    Args:
        series: 原始一维序列。
        n_surrogates: 替代序列数量。
        rng: numpy 随机数生成器, 不传则内部创建。

    Returns:
        形状 (n_surrogates, n) 的替代序列矩阵。
    """
    x = np.asarray(series, dtype=float).ravel()
    n = len(x)
    rng = rng or np.random.default_rng()
    amps = np.abs(np.fft.rfft(x - x.mean()))
    n_coef = len(amps)
    surr = np.empty((n_surrogates, n))
    for k in range(n_surrogates):
        phase = rng.uniform(0.0, 2.0 * np.pi, size=n_coef)
        spec = amps * np.exp(1j * phase)
        spec[0] = amps[0]            # 直流分量保持实数
        if n % 2 == 0:
            spec[-1] = amps[-1]      # 奈奎斯特分量保持实数
        surr[k] = np.fft.irfft(spec, n=n)
    return surr


def sample_entropy(series: np.ndarray, m: int = SAMPEN_M,
                   r_frac: float = SAMPEN_R_FRAC) -> float:
    """样本熵 SampEn(m, r) = -ln(A/B)。

    A = m+1 维模板匹配对数, B = m 维模板匹配对数（Chebyshev 距离 < r,
    r = r_frac * std, 排除自身匹配）。用 KDTree 半径查询实现, O(N log N)。

    Args:
        series: 一维序列。
        m: 模板维数。
        r_frac: 容差系数（相对标准差）。

    Returns:
        样本熵; 无匹配时返回 NaN（表示序列过于规则, 熵为 0 的边界情况）。
    """
    x = np.asarray(series, dtype=float).ravel()
    x = x - x.mean()
    sd = float(x.std())
    if sd == 0.0 or len(x) < m + 2:
        return 0.0
    r = r_frac * sd

    def _count(mdim: int) -> int:
        """统计 mdim 维模板在半径 r 内的匹配对数（Chebyshev 范数）。"""
        tmpl = np.lib.stride_tricks.sliding_window_view(x, mdim)  # (n-mdim+1, mdim)
        pairs = _KDTree(tmpl).query_pairs(r, p=np.inf, output_type='ndarray')
        return int(len(pairs))

    b = _count(m)       # m 维匹配对数
    a = _count(m + 1)   # m+1 维匹配对数
    if b <= 0:
        return float('nan')
    return float(-np.log(max(a, 1) / b))


def surrogate_test(series: np.ndarray, n_surrogates: int = 100, metric: str = 'lyap',
                   method: str = 'fft', seed: Optional[int] = None,
                   dim: Optional[int] = None, tau: Optional[int] = None) -> Dict:
    """Surrogate data 检验: 原始序列混沌指标 vs 替代序列分布。

    Args:
        series: 一维序列。
        n_surrogates: 替代序列数量（默认 100）。
        metric: 检验指标, 'lyap'（Rosenstein 最大 Lyapunov 指数, 默认）或 'sampen'。
        method: 替代序列生成方式, 'fft'（FFT 相位随机化, 默认）或
            'shuffle'（同分布随机洗牌, 只破坏时间顺序、保留幅值分布,
            用于复核分布形状伪影）。
        seed: 随机种子（保证可复现）。
        dim: 嵌入维（不传则由 FNN 自动选择）。
        tau: 延迟（不传则由互信息第一极小自动选择）。

    Returns:
        dict: {'metric', 'value', 'surr_mean', 'surr_std', 'z_score',
               'p_value', 'n_surrogates', 'verdict'}
        - z_score = (原始值 - 替代均值) / 替代标准差;
        - p_value 为双侧经验 p（带伪计数, 无正态假设）;
        - verdict: p>0.05 → RANDOM; p<=0.05 且 z>2 → CHAOTIC/STRUCTURED; 否则 RANDOM。
    """
    x = np.asarray(series, dtype=float).ravel()
    rng = np.random.default_rng(seed)
    tau = mutual_info_first_min(x) if tau is None else int(tau)
    dim = choose_embed_dim(x, tau) if dim is None else int(dim)
    if metric == 'lyap':
        fn = lambda s: max_lyapunov_rosenstein(s, dim, tau)  # noqa: E731
    elif metric == 'sampen':
        fn = lambda s: sample_entropy(s)                     # noqa: E731
    else:
        raise ValueError(f"未知指标: {metric}（可选 'lyap' / 'sampen'）")
    if method == 'fft':
        surr = fft_surrogates(x, n_surrogates, rng)
    elif method == 'shuffle':
        surr = np.array([rng.permutation(x) for _ in range(n_surrogates)])
    else:
        raise ValueError(f"未知替代方式: {method}（可选 'fft' / 'shuffle'）")
    value = float(fn(x))
    vals = np.array([fn(s) for s in surr], dtype=float)
    vals = vals[np.isfinite(vals)]
    if vals.size == 0 or not np.isfinite(value):
        return {'metric': metric, 'value': value, 'surr_mean': float('nan'),
                'surr_std': float('nan'), 'z_score': float('nan'),
                'p_value': float('nan'), 'n_surrogates': n_surrogates,
                'verdict': 'RANDOM'}
    mu, sd = float(vals.mean()), float(vals.std(ddof=1))
    z = (value - mu) / sd if sd > 0 else float('nan')
    # 主 p 值: 正态近似双侧（与 z 一致, 无经验 p 的 n 分辨率下限,
    # 避免小样本 surrogate 时永远判不显著的功效问题）
    p_norm = 2.0 * float(stats.norm.sf(abs(z))) if np.isfinite(z) else float('nan')
    # 附加: 双侧经验 p（加伪计数避免 0/1, 无正态假设, 仅参考）
    p_hi = (np.sum(vals >= value) + 1) / (vals.size + 1)
    p_lo = (np.sum(vals <= value) + 1) / (vals.size + 1)
    p_emp = min(1.0, 2.0 * min(p_hi, p_lo))
    # 判定规则: p>0.05 → RANDOM; p<=0.05 且 z>2 → CHAOTIC/STRUCTURED
    if not np.isfinite(z) or p_norm > ALPHA:
        verdict = 'RANDOM'
    elif z > Z_THRESHOLD:
        verdict = 'CHAOTIC/STRUCTURED'
    else:
        verdict = 'RANDOM'
    return {'metric': metric, 'value': value, 'surr_mean': mu, 'surr_std': sd,
            'z_score': z, 'p_value': p_norm, 'p_empirical': p_emp,
            'n_surrogates': n_surrogates, 'verdict': verdict}


def run_chaos_test(series: np.ndarray, n_surrogates: int = 100,
                   seed: Optional[int] = None) -> Dict:
    """完整混沌检验: 嵌入参数估计 + Lyapunov 双对照 surrogate 检验（主）+ SampEn 对照。

    Args:
        series: 一维时间序列。
        n_surrogates: 替代序列数量（默认 100）。
        seed: 随机种子（保证可复现）。

    Returns:
        dict: {'n', 'embed_dim', 'tau', 'lyap', 'lyap_shuffle', 'sampen',
               'sampen_shuffle', 'verdict', 'note'}
        主指标 Lyapunov 跑两个对照: 'lyap'=FFT 相位随机化（相位结构）,
        'lyap_shuffle'=同分布洗牌（时间结构, 消除分布形状伪影）。
        最终 verdict 要求两个对照都显著（p<=0.05 且 z>2）才判
        CHAOTIC/STRUCTURED; 只要同分布洗牌对照无差异即判 RANDOM。
        SampEn 两个对照并列供交叉验证; note 为中文结论说明。
    """
    x = np.asarray(series, dtype=float).ravel()
    tau = mutual_info_first_min(x)
    dim = choose_embed_dim(x, tau)
    lyap = surrogate_test(x, n_surrogates, metric='lyap', method='fft', seed=seed, dim=dim, tau=tau)
    lyap_sh = surrogate_test(x, n_surrogates, metric='lyap', method='shuffle', seed=seed, dim=dim, tau=tau)
    sampen = surrogate_test(x, n_surrogates, metric='sampen', method='fft', seed=seed, dim=dim, tau=tau)
    sampen_sh = surrogate_test(x, n_surrogates, metric='sampen', method='shuffle', seed=seed, dim=dim, tau=tau)
    # 最终判定: 两个对照都显著才承认结构; 洗牌对照不显著 → 分布伪影, 判随机
    both_significant = (
        lyap['p_value'] <= ALPHA and lyap['z_score'] > Z_THRESHOLD
        and lyap_sh['p_value'] <= ALPHA and lyap_sh['z_score'] > Z_THRESHOLD
    )
    verdict = 'CHAOTIC/STRUCTURED' if both_significant else 'RANDOM'
    fft_note = (
        f"FFT 相位随机化对照: λ_max(原始)={lyap['value']:.4f} vs "
        f"替代 {lyap['surr_mean']:.4f}±{lyap['surr_std']:.4f}, "
        f"z={lyap['z_score']:.2f}, p={lyap['p_value']:.3f}; "
        f"SampEn: z={sampen['z_score']:.2f}, p={sampen['p_value']:.3f}。"
    )
    sh_note = (
        f"同分布洗牌对照: λ_max z={lyap_sh['z_score']:.2f}, p={lyap_sh['p_value']:.3f}; "
        f"SampEn z={sampen_sh['z_score']:.2f}, p={sampen_sh['p_value']:.3f}。"
    )
    if verdict == 'RANDOM':
        if lyap['p_value'] <= ALPHA and lyap_sh['p_value'] > ALPHA:
            note = (fft_note + sh_note
                    + "FFT 对照显著但同分布洗牌对照无差异 → 差异源于分布形状伪影"
                      "（离散整数 vs 连续替代）, 无时间结构证据 → 纯随机。")
        else:
            note = (fft_note + sh_note
                    + "两个对照均无显著差异 → 无确定性混沌结构（纯随机）。")
    else:
        note = (fft_note + sh_note
                + "两个对照均显著 → 疑似确定性结构, 需谨慎解读（建议独立复核）。")
    return {'n': len(x), 'embed_dim': dim, 'tau': tau, 'lyap': lyap,
            'lyap_shuffle': lyap_sh, 'sampen': sampen, 'sampen_shuffle': sampen_sh,
            'verdict': verdict, 'note': note}
