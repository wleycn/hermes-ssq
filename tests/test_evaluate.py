#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""evaluate.py / ml.decode / walk_forward 增强 / feature_engineer 增量 的单元测试（Dev-2）。

运行: .venv/bin/python -m pytest tests/test_evaluate.py -q
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

SSQ = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(SSQ))

from ml.data import load_data
from ml.decode import constrained_decode
from ml.eval.walk_forward import (
    BLUE_RANDOM_EXPECT,
    RED_RANDOM_EXPECT,
    blue_random_baseline,
    random_red_overlap_period,
)
from ml.features.feature_engineer import FeatureEngineer

import evaluate
from evaluate import (
    EvalContext,
    STRATEGIES,
    build_report,
    compare_features,
    load_pg_probs,
    main,
    run_backtest,
    write_report,
)

COMPACT_COLS_BASELINE = 160  # 改动前实测: compact 模式 160 列 / 3487 行


# ================= fixtures =================
@pytest.fixture(scope="module")
def df() -> pd.DataFrame:
    return load_data()


@pytest.fixture(scope="module")
def compact_baseline(df: pd.DataFrame) -> pd.DataFrame:
    """改动前管线回归基线: build_unified_features(mode='compact') 的输出。"""
    return FeatureEngineer().build_unified_features(df, mode="compact")


def _ctx(seed: int = 1, pool_size: int = 12) -> EvalContext:
    return EvalContext(config={"seed": seed, "pool_size": pool_size})


# ================= 1. --smoke 跑通 5 期 =================
def test_smoke_run_5_periods(df: pd.DataFrame, tmp_path: Path):
    out = tmp_path / "smoke.json"
    report = main(["--smoke", "--out", str(out)])
    assert report["n_periods"] == 5
    assert out.exists() and out.with_suffix(".md").exists()


# ================= 2. 报告 schema 字段齐全 =================
def test_report_schema_fields(df: pd.DataFrame):
    report = build_report(df, "freq", horizon=10, train_min=800, ctx=_ctx(),
                          n_trials=50, seed=1)
    top_keys = {"strategy", "kind", "features", "n_periods", "train_min", "run_at",
                "red", "blue", "verdict_overall", "per_period", "notes"}
    assert top_keys <= set(report.keys())
    red_keys = {"mean_hits", "baseline_mean", "baseline_ci95", "delta", "delta_ci95",
                "t_stat", "t_pvalue", "n_ge_50", "significant", "verdict"}
    blue_keys = {"hit_rate", "baseline_rate", "delta", "delta_ci95", "t_pvalue",
                 "significant", "verdict"}
    assert red_keys <= set(report["red"].keys())
    assert blue_keys <= set(report["blue"].keys())
    assert report["n_periods"] == 10
    pp = report["per_period"][0]
    assert {"t", "red_hits", "blue_hit", "pred_reds", "pred_blue",
            "true_reds", "true_blue"} <= set(pp.keys())
    assert len(pp["pred_reds"]) == 6 and len(set(pp["pred_reds"])) == 6
    assert 1 <= pp["pred_blue"] <= 16


# ================= 3. random 策略 mean_hits 落在理论期望 ±0.3 =================
def test_random_mean_within_theory(df: pd.DataFrame):
    bt = run_backtest(df, STRATEGIES["random"]["fn"], horizon=100, train_min=800,
                      ctx=_ctx(), n_trials=50, seed=1)
    mean_hits = float(bt["red_arr"].mean())
    assert RED_RANDOM_EXPECT - 0.3 <= mean_hits <= RED_RANDOM_EXPECT + 0.3, mean_hits
    assert len(bt["red_arr"]) == 100


# ================= 4. verdict 字段取值合法 =================
def test_verdict_values_valid(df: pd.DataFrame):
    report = build_report(df, "hot_cold", horizon=10, train_min=800, ctx=_ctx(),
                          n_trials=50, seed=1)
    assert report["red"]["verdict"] in ("keep", "rollback")
    assert report["blue"]["verdict"] in ("keep", "rollback")
    assert report["verdict_overall"] in ("keep", "rollback")
    assert report["verdict_overall"] == report["red"]["verdict"]
    assert isinstance(report["red"]["significant"], bool)
    assert isinstance(report["blue"]["significant"], bool)


# ================= 5. md/json 双输出可解析 =================
def test_md_json_parseable(df: pd.DataFrame, tmp_path: Path):
    report = build_report(df, "freq", horizon=10, train_min=800, ctx=_ctx(),
                          n_trials=50, seed=1)
    jp, mp = tmp_path / "r.json", tmp_path / "r.md"
    write_report(report, jp)
    write_report(report, mp)
    loaded = json.loads(jp.read_text(encoding="utf-8"))
    assert loaded["strategy"] == "freq"
    assert loaded["n_periods"] == 10
    md = mp.read_text(encoding="utf-8")
    assert "红球指标" in md and "判定" in md and "总体判定" in md


# ================= 6. calc_ac_features 已知样例 =================
def test_calc_ac_features_known_samples():
    fe = FeatureEngineer()
    dfx = pd.DataFrame({
        "Red1": [1, 1], "Red2": [2, 2], "Red3": [3, 4],
        "Red4": [4, 8], "Red5": [5, 16], "Red6": [6, 32],
    })
    out = fe.calc_ac_features(dfx)
    assert out["AC_Value"].tolist() == [0.0, 10.0]
    # 等差序列: 两两差去重 5 个 -> AC=0
    dfy = pd.DataFrame({
        "Red1": [1], "Red2": [7], "Red3": [13], "Red4": [19], "Red5": [25], "Red6": [31],
    })
    assert fe.calc_ac_features(dfy)["AC_Value"].tolist() == [0.0]


# ================= 7. build_unified_features 无回归 + keep_override =================
def test_build_unified_features_no_regression(compact_baseline: pd.DataFrame):
    assert compact_baseline.shape[1] == COMPACT_COLS_BASELINE
    assert "AC_Value" not in compact_baseline.columns


def test_build_unified_features_keep_override(df: pd.DataFrame):
    out = FeatureEngineer().build_unified_features(df, mode="compact",
                                                   keep_override=["AC_Value"])
    assert out.shape[1] == COMPACT_COLS_BASELINE + 1
    assert "AC_Value" in out.columns
    assert out["AC_Value"].between(0, 10).all()


# ================= 8. constrained_decode 合法性 + 退化输入 =================
def test_constrained_decode_valid():
    rng = np.random.default_rng(0)
    probs = rng.dirichlet(np.ones(33))
    reds = constrained_decode(probs, k=6)
    assert len(reds) == 6 and len(set(reds)) == 6
    assert all(1 <= x <= 33 for x in reds)
    odds = sum(1 for x in reds if x % 2 == 1)
    big = sum(1 for x in reds if x > 16)
    assert odds in (2, 3, 4), odds
    assert big in (2, 3, 4), big


def test_constrained_decode_degenerate():
    assert constrained_decode(np.array([])) == []
    assert constrained_decode(np.array([np.nan] * 33)) == list(range(1, 7))
    assert constrained_decode(np.zeros(33)) == list(range(1, 7))
    assert constrained_decode(np.array([0.5, 0.5])) == [1, 2]
    assert constrained_decode(np.array([0.3, 0.7])) == [1, 2]


# ================= 9. walk_forward 增强(向后兼容) =================
def test_walk_forward_new_constants():
    assert abs(RED_RANDOM_EXPECT - 6 * 6 / 33.0) < 1e-12
    assert abs(BLUE_RANDOM_EXPECT - 1 / 16.0) < 1e-12
    assert blue_random_baseline() == 1 / 16.0


def test_random_red_overlap_period(df: pd.DataFrame):
    arr = random_red_overlap_period(df, 800, 809, n_trials=200, seed=1)
    assert arr.shape == (10,)
    assert abs(float(arr.mean()) - RED_RANDOM_EXPECT) < 0.15
    # 与旧接口 random_red_overlap_actual 同口径
    from ml.eval.walk_forward import random_red_overlap_actual
    assert abs(random_red_overlap_actual(df, 800, 809, n_trials=200, seed=1)
               - float(arr.mean())) < 1e-9


# ================= 10. 策略注册表 / CLI 枚举 =================
def test_strategy_registry_complete():
    for name in ("random", "freq", "uniform", "entropy", "hot_cold",
                 "ac", "crf", "diversity", "model:pg"):
        assert name in STRATEGIES


def test_cli_list(tmp_path: Path, capsys):
    res = main(["--list"])
    assert res == {"kind": "list"}
    captured = capsys.readouterr().out
    assert "entropy" in captured and "model:pg" in captured


# ================= 11. compare_features 多特征全量入报告 =================
def test_compare_features_all_in_report(df: pd.DataFrame, tmp_path: Path):
    out = tmp_path / "fv.json"
    multi = main(["--features", "entropy,hot_cold,ac,crf,diversity",
                  "--horizon", "5", "--seed", "1", "--out", str(out)])
    assert multi["kind"] == "feature_validation"
    assert [s["strategy"] for s in multi["strategies"]] == \
        ["entropy", "hot_cold", "ac", "crf", "diversity"]
    for s in multi["strategies"]:
        # 5 特征 × 红/蓝 全量入报告, 不挑拣
        assert "red" in s and "blue" in s
        assert s["verdict_overall"] in ("keep", "rollback")
    assert out.exists()
    loaded = json.loads(out.read_text(encoding="utf-8"))
    assert len(loaded["strategies"]) == 5


# ================= 12. model:pg 依赖 PG(有数据则跑, 无则 graceful skip) =================
def test_model_pg_graceful_or_run(df: pd.DataFrame):
    pg = load_pg_probs()
    ctx = EvalContext(config={"seed": 1})
    if pg is None:
        report = build_report(df, "model:pg", horizon=5, train_min=800, ctx=ctx,
                              n_trials=50, seed=1)
        assert report["n_periods"] == 0
        assert any("策略执行失败" in n for n in report["notes"])
    else:
        ctx.probs = pg
        report = build_report(df, "model:pg", horizon=5, train_min=800, ctx=ctx,
                              n_trials=50, seed=1)
        assert report["n_periods"] == 5
        assert any("冻结" in n for n in report["notes"])
        assert report["per_period"][0]["pred_blue"] is not None
