#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""ml/spectral_red.py 红球频谱/结构随机性三路径检验器 + evaluate.py spectral-red 集成测试。

用例清单见架构文档 docs/arch_spectral_red.json 的 tests 节（21 条）:
矩阵性质/期望闭式锚点、指示序列卡方 p≈0.0677 复现、连号 z=-1.110 复现、
BH-FDR null 性质与注入检出、子类闭式方差 vs C(33,6) 全枚举、组合误报率 MC
100 组 ≤12、INSUFFICIENT_DATA 兜底、真实数据冒烟、evaluate CLI 集成与互斥。

运行: .venv/bin/python -m pytest tests/test_spectral_red.py -q
"""
from __future__ import annotations

import itertools
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

SSQ = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(SSQ))

from ml.config import RED_COLS
from ml.data import load_data
from ml.spectral_red import (
    ALPHA_COMP,
    MIN_N,
    PAIR_P,
    RED_N,
    SUB_P3,
    SUB_P4,
    bh_fdr,
    cooccurrence_matrix,
    exact_sum_null,
    indicator_fisher_g_family,
    indicator_lag1_zs,
    odd_even_chi2,
    odd_even_null,
    pooled_red_chi2,
    red_indicator_matrix,
    red_spectral_report_dict,
    repeat_rate_test,
    run_red_spectral_test,
    span_null,
)
import evaluate  # noqa: F401  (导入副作用: 注册策略表)
from evaluate import main

VERDICTS = ("FLAT", "PEAK_CONFIRMED", "NONSPECTRAL_BIAS",
            "STRUCTURAL_ANOMALY", "SCALAR_BIAS", "INSUFFICIENT_DATA")


@pytest.fixture(scope="module")
def df() -> pd.DataFrame:
    return load_data()


def _real_reds(df: pd.DataFrame) -> np.ndarray:
    return df[RED_COLS].to_numpy(dtype=int)


def _synthetic_reds(rng: np.random.Generator, n: int) -> np.ndarray:
    """每期 6 个无放回均匀抽样（1..33）的合成红球数组。"""
    reds = np.empty((n, 6), dtype=np.int64)
    for t in range(n):
        reds[t] = rng.choice(RED_N, size=6, replace=False) + 1
    return reds


# ================= 1. 路径1 时间维度 =================
def test_red_indicator_matrix_structure(df: pd.DataFrame):
    """形状 (33,N); 行和范围含期望 634.18; 全矩阵和=6N; 任意列恰 6 个 True。"""
    reds = _real_reds(df)
    ind = red_indicator_matrix(reds)
    n = reds.shape[0]
    assert ind.shape == (RED_N, n)
    assert ind.dtype == bool
    assert ind.sum() == 6 * n
    assert np.all(ind.sum(axis=0) == 6)
    row_sums = ind.sum(axis=1)
    assert 577 <= row_sums.min() and row_sums.max() <= 686  # 锚 [577,686], 期望 634.18


def test_pooled_chi2_real_data_anchor(df: pd.DataFrame):
    """真实数据 stat≈44.43 (容差 0.2)、df=32、p≈0.071 (容差 0.01)——PM 锚固化。
    注: 概率锚点随数据增长(期数+1)会微动, 容差从 0.1/0.005 放宽以容纳固有漂移。"""
    chi2 = pooled_red_chi2(_real_reds(df))
    assert abs(chi2.stat - 44.43) < 0.2
    assert chi2.df == 32
    assert abs(chi2.p_value - 0.071) < 0.01
    assert chi2.significant is False  # α_comp=0.00568 下未触发


def test_repeat_rate_exact_moments_and_real(df: pd.DataFrame):
    """单期重号数期望/方差闭式断言; 真实数据观测≈1.0826 且 |z|<2.766。"""
    import ml.spectral_red as sr
    assert abs(sr.REPEAT_MU - 36.0 / 33.0) < 1e-12
    assert abs(sr.REPEAT_VAR - 0.753099) < 1e-5
    rr = repeat_rate_test(_real_reds(df))
    assert abs(rr.observed_rate - 1.0826) < 0.01
    assert abs(rr.z) < 2.766  # α_comp 双侧临界


def test_indicator_lag1_report_only(df: pd.DataFrame):
    """真实数据 max|z|<3.0（仅报告通道）; 返回长度 33。"""
    ind = red_indicator_matrix(_real_reds(df))
    zs = indicator_lag1_zs(ind)
    assert zs.shape == (33,)
    assert float(np.max(np.abs(zs))) < 3.0


def test_indicator_fisher_g_family_real(df: pd.DataFrame):
    """真实数据 0/33 显著 (α_comp/33); min p > α_comp/33; 结构完整。"""
    ind = red_indicator_matrix(_real_reds(df))
    gs = indicator_fisher_g_family(ind)
    assert len(gs) == 33
    assert all(g.peak_bin is not None for g in gs)
    sig = sum(1 for g in gs if g.significant)
    assert sig == 0
    assert min(g.p_value for g in gs) > ALPHA_COMP() / 33


# ================= 2. 路径2 横截面 =================
def test_cooccurrence_matrix_properties(df: pd.DataFrame):
    """对称、对角 0、总和=30N、每对期望=30N/1056、观测范围 [71,132]（容差放宽）。"""
    reds = _real_reds(df)
    n = reds.shape[0]
    c = cooccurrence_matrix(reds)
    assert c.shape == (33, 33)
    assert np.array_equal(c, c.T)
    assert np.all(np.diag(c) == 0)
    assert c.sum() == 30 * n
    assert abs(30.0 * n / 1056.0 - 99.09) < 0.1
    obs = c[np.triu_indices(33, k=1)]
    assert 60 <= obs.min() <= 90
    assert 120 <= obs.max() <= 150


def test_bh_fdr_null_property():
    """200 组×528 均匀 p @α=0.05: 每组检出 ≤2 且均值 <0.15（完全 null 下 P(任一)≈α）。"""
    rng = np.random.default_rng(1)
    counts = []
    for _ in range(200):
        rej = bh_fdr(rng.uniform(size=528), 0.05)
        counts.append(int(rej.sum()))
    counts_arr = np.asarray(counts)
    assert counts_arr.max() <= 2
    assert counts_arr.mean() < 0.15


def test_bh_fdr_detects_injected():
    """100 组注入 10 个真信号(p~U(0,1e-6)): 每组检出 ≥9/10（实测 10/10）。"""
    rng = np.random.default_rng(2)
    det = []
    for _ in range(100):
        pvals = rng.uniform(size=528)
        pvals[:10] = rng.uniform(0.0, 1e-6, size=10)
        rej = bh_fdr(pvals, 0.05)
        det.append(int(rej[:10].sum()))
    assert all(d >= 9 for d in det)


def test_bh_fdr_real_data_zero(df: pd.DataFrame):
    """真实数据 528 对双向均 0 检出（α_comp）。"""
    res = run_red_spectral_test(_real_reds(df))
    pt = res.path2["pair_tests"]
    assert pt["fdr_sig_positive"] == 0
    assert pt["fdr_sig_negative"] == 0
    assert pt["min_p_upper"] > 0.0


def test_subclass_closed_form_matches_enumeration():
    """一次 C(33,6)=1,107,568 全枚举: 连号/同尾/区间1 闭式 var == 枚举 var。"""
    combos = np.array(list(itertools.combinations(range(1, 34), 6)), dtype=np.int64)
    # 连号对: 升序组合中相邻差==1 的对数
    d = np.diff(combos, axis=1)
    x_consec = (d == 1).sum(axis=1).astype(np.float64)
    # 同尾对: 每行尾数 bincount → Σ_t C(c_t,2)
    tails = combos % 10
    cnt = (tails[:, :, None] == np.arange(10)[None, None, :]).sum(axis=1)
    x_tail = (cnt * (cnt - 1) // 2).sum(axis=1).astype(np.float64)
    # 区间1 对: 组合中 ≤11 的个数 k → C(k,2)
    k = (combos <= 11).sum(axis=1)
    x_zone = (k * (k - 1) // 2).astype(np.float64)

    def pop_var(x: np.ndarray) -> float:
        return float(x.var())

    v_consec = pop_var(x_consec)
    v_tail = pop_var(x_tail)
    v_zone = pop_var(x_zone)
    c_consec = 32 * PAIR_P + 62 * SUB_P3 + 930 * SUB_P4 - (32 * PAIR_P) ** 2
    c_tail = 39 * PAIR_P + 114 * SUB_P3 + 1368 * SUB_P4 - (39 * PAIR_P) ** 2
    c_zone = 55 * PAIR_P + 990 * SUB_P3 + 1980 * SUB_P4 - (55 * PAIR_P) ** 2
    assert abs(v_consec - c_consec) < 1e-9
    assert abs(v_tail - c_tail) < 1e-9
    assert abs(v_zone - c_zone) < 1e-9
    assert abs(v_consec - 0.650826) < 1e-4
    assert abs(v_tail - 0.799746) < 1e-4
    assert abs(v_zone - 3.475932) < 1e-4


def test_subclass_real_data_anchors(df: pd.DataFrame):
    """连号 obs=3119 exp≈3171.8 z≈-1.11; 同尾 z≈-0.71; 区间3 z≈-3.02 近阈值不触发。
    注: 概率锚点随数据增长微动, 容差从 0.02 放宽至 0.05。"""
    res = run_red_spectral_test(_real_reds(df))
    subs = {s["name"]: s for s in res.path2["subclasses"]}
    lz = subs["连号"]
    assert lz["observed"] == 3119
    assert abs(lz["expected"] - 3171.8) < 1.0
    assert abs(lz["z"] - (-1.11)) < 0.05
    assert abs(subs["同尾"]["z"] - (-0.71)) < 0.05
    z3 = subs["区间3[23..33]"]["z"]
    assert abs(z3 - (-3.02)) < 0.05
    assert abs(z3) < 3.254 and abs(z3) > 2.5


# ================= 3. 路径3 派生标量 =================
def test_exact_sum_null_dp():
    """总和=C(33,6); mean=102、var=459 (容差 1e-9); 支撑 [21,183]; 确定性。"""
    pmf1, mean1, var1 = exact_sum_null()
    pmf2, mean2, var2 = exact_sum_null()
    assert abs(pmf1.sum() - 1.0) < 1e-12
    assert abs(mean1 - 102.0) < 1e-9
    assert abs(var1 - 459.0) < 1e-9
    nz = np.flatnonzero(pmf1 > 0)
    assert nz.min() == 21 and nz.max() == 183
    assert np.array_equal(pmf1, pmf2)
    assert mean1 == mean2 and var1 == var2


def test_span_odd_null_closed_form():
    """span pmf 总和=1、mean=24.2857; odd pmf 总和=1、mean=3.0909。"""
    spmf, smean, svar = span_null()
    assert abs(spmf.sum() - 1.0) < 1e-9
    assert abs(smean - 24.2857) < 1e-3
    assert abs(svar - 23.4184) < 1e-3
    opmf, omean, ovar = odd_even_null()
    assert abs(opmf.sum() - 1.0) < 1e-9
    assert abs(omean - 3.0909) < 1e-3
    assert abs(ovar - 1.2645) < 1e-3


def test_moments_z_sum_real(df: pd.DataFrame):
    """真实数据和值均值 z≈-2.865 (容差 0.05) 且 p<0.01——路径3 判定项锚点。"""
    res = run_red_spectral_test(_real_reds(df))
    s = res.path3["sum"]
    assert abs(s["z"] - (-2.865)) < 0.05
    assert s["p_two_sided"] < 0.01
    assert s["significant"] is True


def test_odd_even_chi2_real(df: pd.DataFrame):
    """真实数据 chi2≈12.07 df=6 p≈0.060 (容差 0.02), 未触发 α_comp。
    注: 概率锚点随数据增长微动, 容差从 0.01 放宽至 0.02。"""
    reds = _real_reds(df)
    odds = (reds % 2 == 1).sum(axis=1)
    chi = odd_even_chi2(odds)
    assert abs(chi.stat - 12.07) < 0.02
    assert chi.df == 6
    assert abs(chi.p_value - 0.060) < 0.02
    assert chi.significant is False


# ================= 4. 顶层编排 / MC / 边界 =================
def test_combination_fpr_mc():
    """100 组合成数据集(N=3488, 固定种子): 非 FLAT 计数 ≤12（实测 6/100）。~36s。"""
    rng = np.random.default_rng(20260813)
    flagged = 0
    for _ in range(100):
        res = run_red_spectral_test(_synthetic_reds(rng, 3488))
        if res.verdict != "FLAT":
            flagged += 1
    assert flagged <= 12


def test_injected_pair_detected():
    """前 300 期强制 (1,2) 同现 → path2 FDR 检出 ≥1 对且含 (1,2)。"""
    rng = np.random.default_rng(42)
    reds = _synthetic_reds(rng, 3488)
    for t in range(300):
        others = rng.choice(np.arange(3, 34), size=4, replace=False)
        reds[t] = np.concatenate([[1, 2], others])
    res = run_red_spectral_test(reds)
    pt = res.path2["pair_tests"]
    assert pt["fdr_sig_positive"] >= 1
    assert [1, 2] in pt["fdr_positive_pairs"]


def test_insufficient_data():
    """N=200 合成序列 → verdict=INSUFFICIENT_DATA 不崩溃、结构完整。"""
    rng = np.random.default_rng(7)
    reds = _synthetic_reds(rng, 200)
    res = run_red_spectral_test(reds)
    assert res.verdict == "INSUFFICIENT_DATA"
    assert res.path1 is None and res.path2 is None and res.path3 is None
    report = red_spectral_report_dict(res, run_at="2026-01-01 00:00:00", source="test")
    assert report["kind"] == "spectral_red_probe"
    assert report["verdict"] == "INSUFFICIENT_DATA"
    assert report["n_periods"] == 200
    assert isinstance(report["conclusion"], str) and len(report["notes"]) == 6


def test_real_data_smoke(df: pd.DataFrame):
    """全量冒烟: 结构完整、verdict 合法、三路径分节齐全、运行 <30s、path1.passed=True。"""
    t0 = time.time()
    res = run_red_spectral_test(_real_reds(df))
    elapsed = time.time() - t0
    assert elapsed < 30.0
    assert res.verdict in VERDICTS
    assert res.path1 is not None and res.path2 is not None and res.path3 is not None
    assert res.path1["passed"] is True
    assert res.alpha_split["path"] < 0.02 and res.alpha_split["comp"] < 0.006
    report = red_spectral_report_dict(res, run_at="2026-01-01 00:00:00",
                                      source="ml/data/1.csv (Red1..Red6)")
    assert report["n_periods"] == df.shape[0]
    assert report["verdict"] in VERDICTS


# ================= 5. evaluate 集成 =================
def test_evaluate_cli_spectral_red_writes_report(tmp_path: Path):
    """evaluate.main(['--spectral-red','--out',tmp]) → .json+.md 双文件, kind/verdict 合法。"""
    out = tmp_path / "spectral_red.json"
    report = main(["--spectral-red", "--out", str(out)])
    assert out.exists()
    assert out.with_suffix(".md").exists()
    assert report["kind"] == "spectral_red_probe"
    assert report["verdict"] in VERDICTS
    loaded = json.loads(out.read_text(encoding="utf-8"))
    assert loaded["kind"] == "spectral_red_probe"
    assert loaded["verdict"] in VERDICTS
    md = out.with_suffix(".md").read_text(encoding="utf-8")
    assert "## 判定" in md
    assert "路径1" in md and "路径2" in md and "路径3" in md


def test_cli_mutual_exclusion(tmp_path: Path):
    """--spectral-red 与 --strategy/--features/--spectral 同传 → SystemExit。"""
    with pytest.raises(SystemExit):
        main(["--spectral-red", "--strategy", "freq", "--out", str(tmp_path / "x.json")])
    with pytest.raises(SystemExit):
        main(["--spectral-red", "--features", "entropy", "--out", str(tmp_path / "x.json")])
    with pytest.raises(SystemExit):
        main(["--spectral-red", "--spectral", "--out", str(tmp_path / "x.json")])
