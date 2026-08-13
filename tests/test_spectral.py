#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""ml/spectral.py 蓝球频谱平坦性三关检验器 + evaluate.py spectral 集成的单元测试。

用例清单见架构文档 docs/arch_spectral_probe.json 的 tests 节（18 条）:
编码性质/不变性、Fisher's g 精确 p 闭式锚点与 MC 校准、纯周期检出、
Welch 兜底、卡方/自相关零分布、三关 FPR 上限、混噪周期 20/20、
真实数据冒烟、INSUFFICIENT_DATA、evaluate 策略注册表/CLI 集成。

运行: .venv/bin/python -m pytest tests/test_spectral.py -q
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

SSQ = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(SSQ))

from ml.data import load_data
from ml.spectral import (
    BLUE_N,
    GATE_ALPHA,
    MIN_N,
    SUB_ALPHA,
    chi2_uniform_test,
    circular_encode,
    encoding_invariance,
    fisher_g_pvalue,
    fisher_g_test,
    lag_autocorrelation,
    one_hot_series,
    run_three_gate_test,
    spectral_report_dict,
    welch_spectrum,
)

import evaluate
from evaluate import STRATEGIES, build_report, main

VERDICTS = ("FLAT", "PEAK_CONFIRMED", "PEAK_ARTIFACT", "NONSPECTRAL_BIAS",
            "INSUFFICIENT_DATA")


@pytest.fixture(scope="module")
def df() -> pd.DataFrame:
    return load_data()


def _real_blues(df: pd.DataFrame) -> np.ndarray:
    return df["Blue1"].astype(int).to_numpy()


# ================= 1. 编码 =================
def test_circular_encode_unit_circle():
    """|z|=1 逐点; x=1..16 覆盖 16 个 16 次单位根; z(x+8) = -z(x)。"""
    z = circular_encode(np.arange(1, 17))
    assert np.allclose(np.abs(z), 1.0)
    assert len(np.unique(np.round(z, 12))) == 16
    z_low = circular_encode(np.arange(1, 9))
    assert np.allclose(circular_encode(np.arange(9, 17)), -z_low)


def test_circular_encode_rotation_relation():
    """circular_encode((x-1+c)%16+1) == circular_encode(x)·e^(2πi·c/16)。"""
    x = np.arange(1, 17)
    for c in (1, 3, 8):
        rot = ((x - 1 + c) % 16) + 1
        assert np.allclose(circular_encode(rot),
                           circular_encode(x) * np.exp(2j * np.pi * c / 16))


def test_circular_encode_invalid_input():
    with pytest.raises(ValueError):
        circular_encode(np.array([0, 1, 2]))        # 越界 0
    with pytest.raises(ValueError):
        circular_encode(np.array([1, 17]))          # 越界 17
    with pytest.raises(ValueError):
        circular_encode(np.array([1.5, 2.5]))       # 非整数


def test_one_hot_series_structure():
    """形状 (16,N); 每列恰一个 1; 行 j 指示号码 j+1。"""
    rng = np.random.default_rng(0)
    x = rng.integers(1, 17, size=500)
    oh = one_hot_series(x)
    assert oh.shape == (16, 500)
    assert np.allclose(oh.sum(axis=0), 1.0)
    for j in range(1, 17):
        assert np.array_equal(oh[j - 1], (x == j).astype(float))


def test_one_hot_permutation_invariance():
    """one_hot(π(x)) 的行重排 == π·one_hot(x)（π 为 16 置换）。"""
    rng = np.random.default_rng(1)
    x = rng.integers(1, 17, size=800)
    pi = rng.permutation(16) + 1  # 1-based 置换
    assert np.array_equal(one_hot_series(pi[x - 1])[pi - 1], one_hot_series(x))


# ================= 2. Fisher's g 精确 p =================
def test_fisher_g_pvalue_reference_small_m():
    """闭式锚点: m=10, g=0.25 → p≈0.663414（手算值）; g 单调。"""
    p25 = fisher_g_pvalue(0.25, 10)
    assert abs(p25 - 0.663414) < 1e-6, p25
    # g 越大（峰越强）p 越小: p(0.2) > p(0.25) > p(0.3)
    assert fisher_g_pvalue(0.20, 10) > p25 > fisher_g_pvalue(0.30, 10)
    # 边界
    assert fisher_g_pvalue(0.0, 10) == 1.0
    assert fisher_g_pvalue(-0.1, 10) == 1.0
    assert fisher_g_pvalue(0.25, 1) == 1.0
    assert fisher_g_pvalue(1.5, 10) == 1.0


def test_fisher_g_pvalue_matches_mc_null():
    """m=3487 蒙特卡洛 2000 组复白噪声原始周期图: |精确p(emp 0.95 分位 g)-0.05|<0.02。"""
    rng = np.random.default_rng(0)
    N, m = 3488, 3487
    gs = []
    for _ in range(2000):
        z = np.exp(2j * np.pi * rng.random(N))
        P = np.abs(np.fft.fft(z)) ** 2
        Pnz = P[1:]
        gs.append(float(Pnz.max() / Pnz.sum()))
    g_emp95 = float(np.quantile(gs, 0.95))
    p = fisher_g_pvalue(g_emp95, m)
    assert abs(p - 0.05) < 0.02, (g_emp95, p)


def test_fisher_g_detects_pure_periodic():
    """纯周期 x_t=1+((3t) mod 16), N=3488 → peak_bin=654=3N/16, p<1e-9。"""
    N = 3488
    x = 1 + ((3 * np.arange(N)) % 16)
    fg = fisher_g_test(circular_encode(x))
    assert fg.peak_bin == 654
    assert fg.m == N - 1
    assert fg.p_value < 1e-9
    assert fg.significant
    assert fg.implicated_number == 1


# ================= 3. Welch =================
def test_welch_spectrum_shape_and_fallback():
    """N=3488, W=64 → freqs 长 64(复数双边), n_windows=108; N<W 兜底 nperseg=N。"""
    rng = np.random.default_rng(2)
    z = circular_encode(rng.integers(1, 17, size=3488))
    w64, w128 = welch_spectrum(z, windows=(64, 128), overlap=0.5)
    assert w64.window == 64 and len(w64.freqs) == 64 and len(w64.psd) == 64
    assert w64.n_windows == 108
    assert w64.noverlap == 32
    assert w128.n_windows == 53
    assert 1 <= w64.peak_bin <= 63
    small = welch_spectrum(z[:32], windows=(64,), overlap=0.5)
    assert len(small[0].freqs) == 32
    assert small[0].n_windows == 1


# ================= 4. 门1 子检验 =================
def test_chi2_uniform_known_and_random():
    """手工构造 counts 验证 stat; 均匀随机序列 p 不极端小。"""
    s = np.full(1600, 1)  # counts=[1600,0,...,0], E=100 → stat=24000
    r = chi2_uniform_test(s)
    assert r.stat == pytest.approx(24000.0)
    assert r.df == 15
    assert r.p_value < 1e-10
    assert r.significant
    rng = np.random.default_rng(7)
    x = rng.integers(1, 17, size=3488)
    r2 = chi2_uniform_test(x)
    assert r2.p_value > 1e-3
    assert r2.counts.sum() == 3488
    assert r2.expected == pytest.approx(3488 / 16)


def test_lag_autocorrelation_calibrated_null():
    """随机序列 N=4000 → max_z < 3.5（稳健上界）+ 结构字段齐全。"""
    rng = np.random.default_rng(3)
    x = rng.integers(1, 17, size=4000)
    res = lag_autocorrelation(x, max_lag=20, alpha=0.05)
    assert res.max_z < 3.5, res.max_z
    assert res.rhos.shape == (20,)
    assert res.z_scores.shape == (20,)
    assert 1 <= res.max_z_lag <= 20
    assert res.critical_z > 3.0


# ================= 5. 编码不变性 =================
def test_encoding_invariance_rotation_reflection():
    """旋转/反射 g 与 p 硬断言 |Δg|≤1e-9; permutation 记录字段存在。"""
    rng = np.random.default_rng(11)
    x = rng.integers(1, 17, size=2000)
    inv = encoding_invariance(x, seed=0)
    assert inv.rotation["g_delta_max"] <= 1e-9
    assert inv.rotation["p_stable"] is True
    assert inv.reflection["g_delta"] <= 1e-9
    assert inv.stable is True
    assert "g_shift" in inv.permutation
    assert "peak_bin_shift" in inv.permutation


# ================= 6. 三关编排 =================
def test_three_gate_random_fpr_mc():
    """100 组随机序列(N=3488, rng 固定) → non-FLAT ∈ [0, 12]（实测 4/100）。"""
    rng = np.random.default_rng(42)
    non_flat = 0
    for _ in range(100):
        x = rng.integers(1, 17, size=3488)
        if run_three_gate_test(x).verdict != "FLAT":
            non_flat += 1
    assert 0 <= non_flat <= 12, non_flat


def test_three_gate_detects_noisy_periodic():
    """p_inject=0.3 混噪周期序列(20 种子) → PEAK_CONFIRMED ≥ 18/20（实测 20/20）。

    序列取 x_t=1+(t mod 16)（旋转速度 1/16）: 复谱峰 bin=N/16=218, 门3 复核
    confirmed → PEAK_CONFIRMED（实测 20/20）。注意: 3/16 旋转的复谱峰(bin 654)
    与 one-hot 主频(1/16)虽不同频, 但号码 1 的指示谱是周期 16 脉冲串, 其第 3
    谐波恰在 bin 654 有功率, 落入门3 邻域 {654, N-654}±1 → 仍判 PEAK_CONFIRMED
    （统计正确: 确有周期结构; 门3 按同频邻域而非严格同频判定）。
    """
    N = 3488

    def noisy_periodic(seed: int) -> np.ndarray:
        rng = np.random.default_rng(seed)
        x = 1 + (np.arange(N) % 16)
        mask = rng.random(N) < 0.3
        x[mask] = rng.integers(1, 17, size=int(mask.sum()))
        return x

    confirmed = sum(
        run_three_gate_test(noisy_periodic(s)).verdict == "PEAK_CONFIRMED"
        for s in range(20)
    )
    assert confirmed >= 18, confirmed


def test_three_gate_real_data_smoke(df: pd.DataFrame):
    """真实数据全量: 结构完整、verdict ∈ 4 值、运行 <30s。"""
    blues = _real_blues(df)
    t0 = time.monotonic()
    res = run_three_gate_test(blues)
    elapsed = time.monotonic() - t0
    assert elapsed < 30, elapsed
    assert res.verdict in VERDICTS[:4]
    assert res.gate1 is not None and res.gate2 is not None
    assert res.gate3 is not None and res.invariance is not None
    assert res.rolling is not None and res.rolling["n_windows"] > 0
    assert res.gate2["welch"] and res.gate2["fisher_g"].peak_bin is not None


def test_three_gate_insufficient_data():
    """N=200 < MIN_N=500 → verdict='INSUFFICIENT_DATA' 不崩溃。"""
    rng = np.random.default_rng(5)
    x = rng.integers(1, 17, size=200)
    res = run_three_gate_test(x)
    assert res.verdict == "INSUFFICIENT_DATA"
    assert res.gate1 is None and res.gate2 is None and res.invariance is None
    assert "样本量不足" in res.conclusion


def test_spectral_report_dict_schema():
    """spectral_report_dict 输出可 JSON 序列化、schema 字段齐全。"""
    rng = np.random.default_rng(9)
    x = rng.integers(1, 17, size=1500)
    res = run_three_gate_test(x)
    rep = spectral_report_dict(res, run_at="2026-08-13 12:00:00",
                               source="ml/data/1.csv (Blue1)")
    assert rep["kind"] == "spectral_probe"
    assert rep["n_periods"] == 1500
    assert rep["alpha_split"]["gate"] == pytest.approx(GATE_ALPHA(), abs=1e-5)
    assert rep["alpha_split"]["sub"] == pytest.approx(SUB_ALPHA(), abs=1e-5)
    assert rep["verdict"] in VERDICTS
    assert {"gate1", "gate2", "gate3", "invariance", "rolling"} <= set(rep)
    # 全量 JSON 可序列化（np 类型已转 native）
    json.dumps(rep, ensure_ascii=False)


# ================= 7. evaluate 集成 =================
def test_evaluate_strategy_registry_spectral():
    assert "spectral" in STRATEGIES
    assert STRATEGIES["spectral"]["kind"] == "probe"
    assert callable(STRATEGIES["spectral"]["fn"])


def test_strategy_spectral_smoke(df: pd.DataFrame, tmp_path: Path):
    """直接调用返回合法 Prediction; CLI 兼容模式跑通并标注『检验器不预测』。"""
    ctx = evaluate.EvalContext(config={"seed": 1})
    pred = STRATEGIES["spectral"]["fn"](df.iloc[:900], 800, ctx)
    assert len(pred.reds) == 6 and len(set(pred.reds)) == 6
    assert all(1 <= x <= 33 for x in pred.reds)
    assert 1 <= pred.blue <= 16
    assert pred.blues is None
    # build_report 冒烟（extra_notes 由 CLI 分支追加, 直接调用不含）
    report = build_report(df, "spectral", horizon=5, train_min=800, ctx=ctx,
                          n_trials=50, seed=1)
    assert report["n_periods"] == 5
    assert report["kind"] == "probe"
    # CLI 兼容模式: --strategy spectral 走既有 build_report→run_walk_forward
    out = tmp_path / "spectral_strategy.json"
    cli_report = main(["--strategy", "spectral", "--horizon", "5",
                       "--out", str(out)])
    assert cli_report["n_periods"] == 5
    assert any("检验器不预测" in n for n in cli_report["notes"])
    assert out.exists()


def test_cli_spectral_writes_report(tmp_path: Path):
    """evaluate.main(['--spectral','--out',tmp]) → 文件存在, kind/verdict 合法。"""
    out = tmp_path / "spectral.json"
    report = main(["--spectral", "--out", str(out)])
    assert out.exists()
    assert report["kind"] == "spectral_probe"
    assert report["verdict"] in VERDICTS
    loaded = json.loads(out.read_text(encoding="utf-8"))
    assert loaded["kind"] == "spectral_probe"
    assert loaded["verdict"] in VERDICTS


def test_cli_spectral_mutually_exclusive(tmp_path: Path):
    """--spectral 与 --strategy/--features 互斥 → SystemExit。"""
    with pytest.raises(SystemExit):
        main(["--spectral", "--strategy", "freq", "--out", str(tmp_path / "x.json")])
    with pytest.raises(SystemExit):
        main(["--spectral", "--features", "entropy", "--out", str(tmp_path / "x.json")])
