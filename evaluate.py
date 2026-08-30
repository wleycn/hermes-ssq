#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""双色球策略回测标准化 + 5 项特征增量验证（P1 / Dev-2）。

统一策略抽象 -> walk-forward 回测（复用 ml/eval/walk_forward.py 骨架零改动）
-> 蒙特卡洛基线（N=1000）-> 三指标标准化报告（JSON/MD）+ 显著性判定。

用法:
  python evaluate.py --list
  python evaluate.py --strategy entropy --features entropy --horizon 100 --out analysis/results/report_entropy.md
  python evaluate.py --features ac,entropy,hot_cold,crf,diversity --horizon 100 --out analysis/results/feature_validation.json
  python evaluate.py --strategy model:pg --horizon 50 --out analysis/results/report_model_pg.json
  python evaluate.py --smoke            # horizon=5 链路冒烟
  python evaluate.py --train-min 800 --n-trials 1000 --seed 0 --pool-size 12

说明:
- 特征策略为纯 pandas 打分->选号, 每期秒级, 跑完整 walk-forward。
- model:pg 不重训: 冻结 PG 最新概率作每期固定概率源, 近期参考回测, 结果仅参考。
- 显著性判定(Q3 裁定): significant = mean_hits > baseline_mean + 1.96*SE 且 n>=50;
  verdict: 红球 significant -> keep, 否则 rollback; verdict_overall = 红球判定。
"""
from __future__ import annotations

import argparse
import itertools
import json
import sys
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Callable, Dict, List, NamedTuple, Optional, Tuple

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent
DC_SSQ = Path("/home/hermes/workspace/data-center/ssq")  # 产出真源 2026-08-30 迁移
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from ml.config import RED_COLS, BLUE_COLS  # 只读
from ml.data import load_data
from ml.decode import constrained_decode
from ml.eval.walk_forward import (
    BLUE_RANDOM_EXPECT,
    RED_RANDOM_EXPECT,
    random_red_overlap_period,
    run_walk_forward,
)
from ml.spectral import (
    circular_encode,
    fisher_g_test,
    run_three_gate_test,
    spectral_report_dict,
)
from ml.spectral_red import (
    red_spectral_report_dict,
    run_red_spectral_test,
)

# ================= 评测配置（未来可合并入 ml/config.py） =================
# 特征开关（生产默认）。CLI --features 显式请求某特征时, 本次评测运行临时启用对应开关。
FEATURE_TOGGLES: Dict[str, bool] = {
    "entropy": True,
    "hot_cold": False,
    "ac": False,
    "crf": False,
    "diversity": False,
    "prime_composite": False,
}

# 特征开关 -> build_unified_features keep_override 追加白名单列（特征管线侧联动）
FEATURE_WHITELIST_COLS: Dict[str, List[str]] = {
    "entropy": ["Entropy"],
    "hot_cold": ["Hot_Count", "Cold_Count", "Hot_Cold_Ratio"],
    "ac": ["AC_Value"],
    "crf": [],
    "diversity": [],
    "prime_composite": ["Prime_Count", "Prime_Ratio"],
}

EVAL_CONFIG: Dict = {
    "train_min": 800,
    "horizon": 100,
    "n_trials": 1000,      # 蒙特卡洛基线抽样次数（与 random_red_overlap_actual 默认统一 N）
    "seed": 1,
    "pool_size": 12,       # ac/diversity 策略候选池大小
    "prime_lambda": 1.5,   # prime_composite 质数偏好权重(>1 偏好质数, <1 偏好合数)
    "red_theory_mean": RED_RANDOM_EXPECT,
    "blue_theory_rate": BLUE_RANDOM_EXPECT,
    "sig_alpha": 1.96,     # 显著性阈值系数
    "sig_min_periods": 50, # 显著性最少期数
}


# ================= 统一策略抽象 =================
class Prediction(NamedTuple):
    reds: List[int]
    blue: int
    blues: Optional[List[int]] = None


@dataclass
class EvalContext:
    """策略执行上下文: 冻结概率源 + 配置 + 跨期状态。"""
    probs: Optional[dict] = None          # {"red": np.array(33), "blue": np.array(16)}
    config: dict = field(default_factory=dict)
    state: dict = field(default_factory=dict)  # 跨期状态(如 diversity 已选注)


Strategy = Callable[[pd.DataFrame, int, EvalContext], Prediction]


# ================= 通用小工具 =================
def _rng(ctx: EvalContext, t: int) -> np.random.Generator:
    """每期独立随机流: 种子 = base_seed + t, 避免各期重复同一序列。"""
    base = int(ctx.config.get("seed", EVAL_CONFIG["seed"]))
    return np.random.default_rng(base + t)


def _red_freq(df_trunc: pd.DataFrame, window: int) -> np.ndarray:
    """截至 t-1 的最近 window 期红球频次计数(1..33)。"""
    reds = df_trunc[RED_COLS].astype(int).values[-window:]
    counts = np.zeros(33, dtype=np.float64)
    for row in reds:
        for x in row:
            if 1 <= x <= 33:
                counts[x - 1] += 1
    return counts


def _blue_freq(df_trunc: pd.DataFrame, window: int) -> np.ndarray:
    """截至 t-1 的最近 window 期蓝球频次计数(1..16)。"""
    blues = df_trunc[BLUE_COLS[0]].astype(int).values[-window:]
    return np.bincount(blues, minlength=17)[1:17].astype(np.float64)


def _top_red(counts: np.ndarray, k: int = 6) -> List[int]:
    """按频次降序取前 k 个红球(升序返回号码)。"""
    return sorted((np.argsort(-counts)[:k] + 1).tolist())


def _top_blue(counts: np.ndarray) -> int:
    """按频次取蓝球 argmax(1..16)。"""
    return int(np.argmax(counts)) + 1


def _binary_entropy(p: float) -> float:
    """二元熵 H(p) = -p log2 p - (1-p) log2(1-p)。"""
    p = float(np.clip(p, 1e-10, 1 - 1e-10))
    return -(p * np.log2(p) + (1 - p) * np.log2(1 - p))


def ac_value(nums: List[int]) -> int:
    """AC 值 = 6 号两两差绝对值去重数 - 5, 范围 0~10, 越分散越大。"""
    diffs = {abs(int(a) - int(b)) for i, a in enumerate(nums) for b in nums[i + 1:]}
    return len(diffs) - 5


def _prob_source(df_trunc: pd.DataFrame, ctx: EvalContext) -> np.ndarray:
    """概率源: 优先 ctx.probs 冻结概率, 否则近 200 期频次概率。"""
    if ctx.probs is not None and "red" in ctx.probs:
        p = np.asarray(ctx.probs["red"], dtype=np.float64)
        if p.sum() > 0:
            return p / p.sum()
    counts = _red_freq(df_trunc, 200)
    return counts / max(counts.sum(), 1e-9)


# ================= 策略实现（全部签名 (df_trunc, t, ctx) -> Prediction） =================
def strategy_random(df_trunc: pd.DataFrame, t: int, ctx: EvalContext) -> Prediction:
    """random: 每期 MC 抽多次的基线统计器（单注随机, 供回测与随机基线对照）。"""
    rng = _rng(ctx, t)
    reds = sorted(rng.choice(range(1, 34), size=6, replace=False).tolist())
    blue = int(rng.choice(range(1, 17), size=1)[0])
    return Prediction(reds=reds, blue=blue)


def strategy_uniform(df_trunc: pd.DataFrame, t: int, ctx: EvalContext) -> Prediction:
    """uniform: 纯随机 6 红 + 随机 1 蓝（与 random 同机制, 语义上为单次随机注单）。"""
    return strategy_random(df_trunc, t, ctx)


def strategy_freq(df_trunc: pd.DataFrame, t: int, ctx: EvalContext) -> Prediction:
    """freq(基线): 近 200 期频次 top6 红 + 近 50 期频次 top1 蓝。"""
    reds = _top_red(_red_freq(df_trunc, 200), 6)
    blue = _top_blue(_blue_freq(df_trunc, 50))
    return Prediction(reds=reds, blue=blue)


def strategy_entropy(df_trunc: pd.DataFrame, t: int, ctx: EvalContext) -> Prediction:
    """entropy: 每号近 50 期出现位图的二元熵, 取熵最低 6 号(最可预测), 并列按近期频次。"""
    window = 50
    red_f = _red_freq(df_trunc, window)
    red_p = red_f / window
    ent = np.array([_binary_entropy(p) for p in red_p])
    # lexsort: 主键熵升序, 次键近期频次降序(并列按频次高者优先)
    order = np.lexsort((-red_f, ent))
    reds = sorted((order[:6] + 1).tolist())

    blue_f = _blue_freq(df_trunc, window)
    blue_p = blue_f / window
    ent_b = np.array([_binary_entropy(p) for p in blue_p])
    order_b = np.lexsort((-blue_f, ent_b))
    blue = int(order_b[0] + 1)
    return Prediction(reds=reds, blue=blue)


def strategy_hot_cold(df_trunc: pd.DataFrame, t: int, ctx: EvalContext) -> Prediction:
    """hot_cold: score = 近200期频次 - 近800期频次(冷号补位/均值回归), top6; 蓝=近50期top1。"""
    score = _red_freq(df_trunc, 200) - _red_freq(df_trunc, 800)
    reds = _top_red(score, 6)
    blue = _top_blue(_blue_freq(df_trunc, 50))
    return Prediction(reds=reds, blue=blue)


def strategy_ac(df_trunc: pd.DataFrame, t: int, ctx: EvalContext) -> Prediction:
    """ac: 候选池=近200期频次 top pool_size, 穷举 C(pool,6) 取 AC 值最大者; 蓝球同法取池内最热。"""
    pool_size = int(ctx.config.get("pool_size", EVAL_CONFIG["pool_size"]))
    red_f = _red_freq(df_trunc, 200)
    pool_reds = (np.argsort(-red_f)[:pool_size] + 1).tolist()
    best: Optional[List[int]] = None
    best_ac, best_sum = -1, -1.0
    for combo in itertools.combinations(pool_reds, 6):
        ac = ac_value(list(combo))
        key = float(sum(red_f[x - 1] for x in combo))
        if ac > best_ac or (ac == best_ac and key > best_sum):
            best, best_ac, best_sum = sorted(combo), ac, key
    blue_f = _blue_freq(df_trunc, 50)
    pool_blues = (np.argsort(-blue_f)[:pool_size] + 1).tolist()
    blue = max(pool_blues, key=lambda x: blue_f[x - 1])
    return Prediction(reds=best or _top_red(red_f, 6), blue=int(blue))


PRIME_NUMBERS = frozenset({2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31})  # 1..33 内 11 个质数


def strategy_prime_composite(df_trunc: pd.DataFrame, t: int, ctx: EvalContext) -> Prediction:
    """prime_composite: score = 近200期频次 x 质数偏好权重, top6; 蓝=近50期top1。

    质数集合(1..33)共 11 个, 理论占比 11/33。lambda>1 偏好质数(质多方向),
    lambda<1 偏好合数(合多方向)。默认 1.5 质多方向; 回测不显著可换 lambda 对照。
    """
    lam = float(ctx.config.get("prime_lambda", EVAL_CONFIG.get("prime_lambda", 1.5)))
    score = _red_freq(df_trunc, 200).astype(np.float64)
    for i in range(33):
        if (i + 1) in PRIME_NUMBERS:
            score[i] *= lam
    reds = _top_red(score, 6)
    blue = _top_blue(_blue_freq(df_trunc, 50))
    return Prediction(reds=reds, blue=blue)


def strategy_crf(df_trunc: pd.DataFrame, t: int, ctx: EvalContext) -> Prediction:
    """crf: 概率源=近200期频次概率(或 ctx.probs 冻结 PG 概率), 约束 beam 解码(ml/decode.py)。"""
    probs = _prob_source(df_trunc, ctx)
    reds = constrained_decode(probs, k=6)
    if ctx.probs is not None and "blue" in ctx.probs:
        bp = np.asarray(ctx.probs["blue"], dtype=np.float64)
        blue = int(np.argmax(bp) + 1) if bp.sum() > 0 else _top_blue(_blue_freq(df_trunc, 50))
    else:
        blue = _top_blue(_blue_freq(df_trunc, 50))
    return Prediction(reds=reds, blue=blue)


def strategy_diversity(df_trunc: pd.DataFrame, t: int, ctx: EvalContext) -> Prediction:
    """diversity: 温度0.6 采样 K=100 候选注, 贪心选注使与已选注重合<=2(跨期状态)。"""
    temp, k_samples = 0.6, 100
    probs = _prob_source(df_trunc, ctx)
    rng = _rng(ctx, t)
    p = np.power(probs, 1.0 / temp)
    p = p / p.sum()
    candidates: List[List[int]] = []
    for _ in range(k_samples):
        picks = rng.choice(33, size=6, replace=False, p=p) + 1
        picks = sorted(picks.tolist())
        odds = sum(1 for x in picks if x % 2 == 1)
        big = sum(1 for x in picks if x > 16)
        if odds in (2, 3, 4) and big in (2, 3, 4):
            candidates.append(picks)
    if not candidates:
        # 无满足约束候选: 退回概率 top6
        candidates = [sorted((np.argsort(-probs)[:6] + 1).tolist())]
    chosen = ctx.state.setdefault("diversity_chosen", [])

    def _overlap(a: List[int], b: List[int]) -> int:
        return len(set(a) & set(b))

    if not chosen:
        best = candidates[0]
    else:
        best, best_score = candidates[0], -10 ** 9
        for cand in candidates:
            max_ov = max(_overlap(cand, c) for c in chosen)
            score = 1.0 if max_ov <= 2 else -float(max_ov)
            if score > best_score:
                best, best_score = cand, score
    chosen.append(best)

    if ctx.probs is not None and "blue" in ctx.probs:
        bp = np.asarray(ctx.probs["blue"], dtype=np.float64)
        blue = int(np.argmax(bp) + 1) if bp.sum() > 0 else _top_blue(_blue_freq(df_trunc, 50))
    else:
        blue = _top_blue(_blue_freq(df_trunc, 50))
    return Prediction(reds=best, blue=blue)


def _sample_red_weighted(red_probs: np.ndarray, rng: np.random.Generator,
                         temperature: float = 0.6) -> List[int]:
    """温度加权抽样 6 红(约束奇偶/大小比), 200 次重试后退回 top6。"""
    p = np.power(np.maximum(red_probs, 0.0), 1.0 / temperature)
    if p.sum() <= 0:
        p = np.ones_like(p)
    p = p / p.sum()
    for _ in range(200):
        picks = rng.choice(33, size=6, replace=False, p=p) + 1
        odds = sum(1 for x in picks if x % 2 == 1)
        big = sum(1 for x in picks if x > 16)
        if odds in (2, 3, 4) and big in (2, 3, 4):
            return sorted(picks.tolist())
    return sorted((np.argsort(-red_probs)[:6] + 1).tolist())


def _sample_blue_weighted(blue_probs: np.ndarray, rng: np.random.Generator,
                          temperature: float = 0.7) -> int:
    """温度加权抽样 1 蓝。"""
    p = np.power(np.maximum(blue_probs, 0.0), 1.0 / temperature)
    if p.sum() <= 0:
        p = np.ones_like(p)
    p = p / p.sum()
    return int(rng.choice(16, size=1, p=p)[0] + 1)


def strategy_model_pg(df_trunc: pd.DataFrame, t: int, ctx: EvalContext) -> Prediction:
    """model:pg: ctx.probs=PG 冻结概率; 红球温度0.6 采样, 蓝球温度0.7 采样。"""
    if ctx.probs is None or "red" not in ctx.probs or "blue" not in ctx.probs:
        raise RuntimeError("PG 冻结概率不可用(model:pg 策略需要 ctx.probs)")
    red_probs = np.asarray(ctx.probs["red"], dtype=np.float64)
    blue_probs = np.asarray(ctx.probs["blue"], dtype=np.float64)
    rng = _rng(ctx, t)
    reds = _sample_red_weighted(red_probs, rng, temperature=0.6)
    blue = _sample_blue_weighted(blue_probs, rng, temperature=0.7)
    return Prediction(reds=reds, blue=blue)


def strategy_spectral(df_trunc: pd.DataFrame, t: int, ctx: EvalContext) -> Prediction:
    """spectral(probe): 红球=纯随机 6 红; 蓝球=近窗复谱 Fisher's g 谱峰相位外推, 无显著峰则随机。

    检验器不预测——本策略输出仅用于 walk-forward 谱结构诊断对照, 不构成下注建议。
    """
    window = int(ctx.config.get("spectral_window", 128))
    reds = strategy_random(df_trunc, t, ctx).reds
    blues = df_trunc[BLUE_COLS[0]].astype(int).to_numpy()[-window:]
    if blues.size >= 2:
        fg = fisher_g_test(circular_encode(blues))
        if fg.significant and fg.implicated_number is not None:
            blue = int(fg.implicated_number)
        else:
            blue = int(_rng(ctx, t).choice(range(1, 17), size=1)[0])
    else:
        blue = int(_rng(ctx, t).choice(range(1, 17), size=1)[0])
    return Prediction(reds=reds, blue=blue)


# ================= 策略注册表 =================
STRATEGIES: Dict[str, Dict] = {
    "random":    {"fn": strategy_random,    "kind": "baseline", "features": []},
    "uniform":   {"fn": strategy_uniform,   "kind": "baseline", "features": []},
    "freq":      {"fn": strategy_freq,      "kind": "baseline", "features": []},
    "entropy":   {"fn": strategy_entropy,   "kind": "feature",  "features": ["entropy"]},
    "hot_cold":  {"fn": strategy_hot_cold,  "kind": "feature",  "features": ["hot_cold"]},
    "ac":        {"fn": strategy_ac,        "kind": "feature",  "features": ["ac"]},
    "crf":       {"fn": strategy_crf,       "kind": "feature",  "features": ["crf"]},
    "diversity": {"fn": strategy_diversity, "kind": "feature",  "features": ["diversity"]},
    "prime_composite": {"fn": strategy_prime_composite, "kind": "feature", "features": ["prime_composite"]},
    "model:pg":  {"fn": strategy_model_pg,  "kind": "model",    "features": []},
    "spectral":  {"fn": strategy_spectral,  "kind": "probe",    "features": []},
}


# ================= PG 冻结概率（model:pg 用, 只读 select_numbers） =================
def load_pg_probs() -> Optional[dict]:
    """读取 PG 最新一次 run 的集成概率。失败返回 None(调用方 graceful skip)。"""
    try:
        import psycopg
        import select_numbers as sn  # 只读: 仅调用 load_latest_probs
        conn = psycopg.connect(**sn.PG)
        try:
            red_mean, blue_mean, run_at, models = sn.load_latest_probs(conn)
        finally:
            conn.close()
        return {"red": red_mean, "blue": blue_mean, "run_at": str(run_at), "models": models}
    except Exception as e:  # noqa: BLE001 - 任何 PG 问题都 graceful skip
        return None


# ================= 回测与统计 =================
def run_backtest(df: pd.DataFrame, strategy: Strategy, horizon: int, train_min: int,
                 ctx: EvalContext, n_trials: int, seed: int) -> Dict:
    """走 walk_forward 骨架(零改动复用)执行单策略回测, 返回明细+统计数组。"""
    # predict_fn 直调策略: (model, df_trunc, t) -> (pred_reds, pred_blue)
    def predict_fn(_model, df_trunc: pd.DataFrame, t: int) -> Tuple[List[int], int]:
        pred = strategy(df_trunc, t, ctx)
        return pred.reds, pred.blue

    res = run_walk_forward(
        model_factory=lambda d: None,
        df=df, train_min=train_min, horizon=horizon, step=1,
        predict_fn=predict_fn,
    )
    # 骨架 end = min(n-1, train_min+horizon) 会多出 1 期, 截断到 horizon 期
    details = res["details"][:horizon]
    n = len(details)
    red_arr = np.array([d["overlap"] for d in details], dtype=np.float64)
    blue_arr = np.array([d["blue_hit"] for d in details], dtype=np.float64)
    start, end = train_min, train_min + n - 1
    mc_arr = random_red_overlap_period(df, start, end, n_trials=n_trials, seed=seed) if n else np.array([])
    return {"details": details, "red_arr": red_arr, "blue_arr": blue_arr,
            "mc_arr": mc_arr, "start": start, "end": end}


def _ttest_1samp(arr: np.ndarray, popmean: float) -> Tuple[Optional[float], Optional[float]]:
    """scipy ttest_1samp, 退化输入(样本<2 或零方差)返回 (None, None) 不崩溃。"""
    from scipy.stats import ttest_1samp
    if len(arr) < 2 or arr.std(ddof=1) == 0:
        return None, None
    try:
        t_stat, p_value = ttest_1samp(arr, popmean)
        return float(t_stat), float(p_value)
    except Exception:  # noqa: BLE001
        return None, None


def _ci_mean(arr: np.ndarray, alpha: float = 1.96) -> List[Optional[float]]:
    """均值 95% CI = mean ± 1.96*SE。"""
    if len(arr) < 2:
        return [None, None]
    se = float(arr.std(ddof=1) / np.sqrt(len(arr)))
    m = float(arr.mean())
    return [m - alpha * se, m + alpha * se]


def build_report(df: pd.DataFrame, strategy_name: str, horizon: int, train_min: int,
                 ctx: EvalContext, n_trials: int, seed: int,
                 features: Optional[List[str]] = None,
                 extra_notes: Optional[List[str]] = None,
                 run_at: Optional[str] = None,
                 imported_comparison: Optional[List[Dict]] = None) -> Dict:
    """三指标标准化报告(schema 见架构规格)。"""
    meta = STRATEGIES[strategy_name]
    features = list(features) if features is not None else list(meta["features"])
    notes: List[str] = list(extra_notes or [])
    notes.append("近似回测: 冻结概率源, 非滚动重训, 结果仅参考") if strategy_name == "model:pg" else None
    notes.append(f"蒙特卡洛基线: 每期 {n_trials} 次随机注单(seed={seed}); "
                 f"红球理论期望 {RED_RANDOM_EXPECT:.4f}, 蓝球理论命中率 {BLUE_RANDOM_EXPECT:.4f}")

    try:
        bt = run_backtest(df, meta["fn"], horizon, train_min, ctx, n_trials, seed)
    except RuntimeError as e:
        # 策略执行失败(如 PG 不可用): graceful skip, 空统计 + notes 标注
        return {
            "strategy": strategy_name, "kind": meta["kind"], "features": features,
            "n_periods": 0, "train_min": train_min, "run_at": run_at or _now(),
            "red": {"mean_hits": None, "baseline_mean": None, "baseline_ci95": [None, None],
                    "delta": None, "delta_ci95": [None, None], "t_stat": None, "t_pvalue": None,
                    "n_ge_50": False, "significant": False, "verdict": "rollback"},
            "blue": {"hit_rate": None, "baseline_rate": BLUE_RANDOM_EXPECT, "delta": None,
                     "delta_ci95": [None, None], "t_pvalue": None, "significant": False,
                     "verdict": "rollback"},
            "verdict_overall": "rollback",
            "per_period": [],
            "notes": notes + [f"策略执行失败, 已跳过: {e}"],
            **({"imported_comparison": imported_comparison} if imported_comparison else {}),
        }

    n = len(bt["red_arr"])
    red_arr, blue_arr, mc_arr = bt["red_arr"], bt["blue_arr"], bt["mc_arr"]
    sig_min = int(EVAL_CONFIG["sig_min_periods"])

    # ---- 红球 ----
    mean_hits = float(red_arr.mean()) if n else None
    baseline_mean = float(mc_arr.mean()) if len(mc_arr) else None
    baseline_ci95 = _ci_mean(mc_arr)
    d = red_arr - mc_arr if n and len(mc_arr) == n else None
    delta = float(d.mean()) if d is not None else None
    delta_ci95 = _ci_mean(d) if d is not None else [None, None]
    t_stat, t_pvalue = _ttest_1samp(
        red_arr, baseline_mean if baseline_mean is not None else RED_RANDOM_EXPECT
    ) if n else (None, None)
    se = float(red_arr.std(ddof=1) / np.sqrt(n)) if n > 1 else float("inf")
    n_ge_50 = n >= sig_min
    red_sig = bool(n_ge_50 and mean_hits is not None and baseline_mean is not None
                   and mean_hits > baseline_mean + EVAL_CONFIG["sig_alpha"] * se)
    red_verdict = "keep" if red_sig else "rollback"

    # ---- 蓝球 ----
    hit_rate = float(blue_arr.mean()) if n else None
    baseline_rate = BLUE_RANDOM_EXPECT
    blue_delta = (hit_rate - baseline_rate) if hit_rate is not None else None
    blue_se = float(np.sqrt(hit_rate * (1 - hit_rate) / n)) if n and hit_rate is not None else None
    blue_ci = ([blue_delta - 1.96 * blue_se, blue_delta + 1.96 * blue_se]
               if blue_delta is not None and blue_se is not None else [None, None])
    _, blue_t_pvalue = _ttest_1samp(blue_arr, baseline_rate) if n else (None, None)
    blue_sig = bool(n_ge_50 and hit_rate is not None and blue_se is not None
                    and hit_rate > baseline_rate + EVAL_CONFIG["sig_alpha"] * blue_se)
    blue_verdict = "keep" if blue_sig else "rollback"

    per_period = [
        {"t": d["t"], "red_hits": d["overlap"], "blue_hit": d["blue_hit"],
         "pred_reds": d["pred_reds"], "pred_blue": d["pred_blue"],
         "true_reds": d["true_reds"], "true_blue": d["true_blue"]}
        for d in bt["details"]
    ]

    report = {
        "strategy": strategy_name, "kind": meta["kind"], "features": features,
        "n_periods": n, "train_min": train_min, "run_at": run_at or _now(),
        "red": {"mean_hits": mean_hits, "baseline_mean": baseline_mean,
                "baseline_ci95": baseline_ci95, "delta": delta, "delta_ci95": delta_ci95,
                "t_stat": t_stat, "t_pvalue": t_pvalue, "n_ge_50": n_ge_50,
                "significant": red_sig, "verdict": red_verdict},
        "blue": {"hit_rate": hit_rate, "baseline_rate": baseline_rate, "delta": blue_delta,
                 "delta_ci95": blue_ci, "t_pvalue": blue_t_pvalue, "significant": blue_sig,
                 "verdict": blue_verdict},
        "verdict_overall": red_verdict,
        "per_period": per_period,
        "notes": notes,
    }
    if imported_comparison:
        report["imported_comparison"] = imported_comparison
    return report


def _now() -> str:
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def _jsonable(obj):
    if isinstance(obj, dict):
        return {k: _jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_jsonable(v) for v in obj]
    if isinstance(obj, np.generic):
        return obj.item()
    return obj


# ================= 输出 =================
def write_report(report: Dict, out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    if out_path.suffix.lower() == ".md":
        kind = report.get("kind")
        if kind == "spectral_red_probe":
            md = render_md_spectral_red(report)
        elif kind == "spectral_probe":
            md = render_md_spectral(report)
        else:
            md = render_md(report)
        out_path.write_text(md, encoding="utf-8")
    else:
        out_path.write_text(json.dumps(_jsonable(report), ensure_ascii=False, indent=2),
                            encoding="utf-8")


def render_md(report: Dict) -> str:
    """单策略报告 -> Markdown 三指标表格 + 判定 + 建议。"""
    r, b = report["red"], report["blue"]
    lines = [
        f"# 策略回测报告: {report['strategy']}",
        "",
        "| 属性 | 值 |",
        "|---|---|",
        f"| 类型(kind) | {report['kind']} |",
        f"| 特征 | {', '.join(report['features']) if report['features'] else '-'} |",
        f"| 回测期数 | {report['n_periods']} |",
        f"| train_min | {report['train_min']} |",
        f"| 运行时间 | {report['run_at']} |",
        "",
        "## 红球指标",
        "| 指标 | 值 |",
        "|---|---|",
        f"| 平均命中 (mean_hits) | {_fmt(r['mean_hits'])} |",
        f"| 随机基线均值 (MC) | {_fmt(r['baseline_mean'])} |",
        f"| 基线 95% CI | {_fmt_ci(r['baseline_ci95'])} |",
        f"| Δ (delta) | {_fmt(r['delta'])} |",
        f"| Δ 95% CI | {_fmt_ci(r['delta_ci95'])} |",
        f"| t 统计量 / p 值 | {_fmt(r['t_stat'])} / {_fmt(r['t_pvalue'])} |",
        f"| n ≥ 50 | {r['n_ge_50']} |",
        f"| 显著性 (> 基线+1.96SE 且 n≥50) | {r['significant']} |",
        f"| 判定 | **{r['verdict']}** |",
        "",
        "## 蓝球指标",
        "| 指标 | 值 |",
        "|---|---|",
        f"| 命中率 (hit_rate) | {_fmt(b['hit_rate'])} |",
        f"| 随机基线 (1/16) | {_fmt(b['baseline_rate'])} |",
        f"| Δ (delta) | {_fmt(b['delta'])} |",
        f"| Δ 95% CI | {_fmt_ci(b['delta_ci95'])} |",
        f"| t 检验 p 值 | {_fmt(b['t_pvalue'])} |",
        f"| 显著性 (> 基线+1.96SE 且 n≥50) | {b['significant']} |",
        f"| 判定 | **{b['verdict']}** |",
        "",
        "## 总体判定",
        f"- verdict_overall: **{report['verdict_overall']}**",
        f"- 建议: {_suggestion(report)}",
        "",
        "## 附注",
    ] + [f"- {note}" for note in report["notes"]]

    if "imported_comparison" in report:
        lines += ["", "## 历史对比(imported: analysis/results/comparison_table.csv)", "",
                  "| model | mode | red_mean_overlap | red_hit_ge3 | blue_top1_acc | n_periods |",
                  "|---|---|---|---|---|---|"]
        for row in report["imported_comparison"]:
            lines.append(f"| {row.get('model', '-')} | {row.get('mode', '-')} | "
                         f"{row.get('red_mean_overlap', '-')} | {row.get('red_hit_ge3', '-')} | "
                         f"{row.get('blue_top1_acc', '-')} | {row.get('n_periods', '-')} |")
    lines.append("")
    return "\n".join(lines)


def _fmt(v) -> str:
    if v is None:
        return "-"
    if isinstance(v, float):
        return f"{v:.4f}"
    return str(v)


def render_md_spectral(report: Dict) -> str:
    """频谱探针报告 -> Markdown: 逐门表格 + verdict 加粗 + conclusion 段落 + notes。

    报告 schema 见架构文档 spectral_probe; INSUFFICIENT_DATA 时各门为 None 显示 "-"。
    """
    lines = [
        "# 蓝球频谱平坦性检验报告（spectral probe）",
        "",
        "| 属性 | 值 |",
        "|---|---|",
        f"| 数据源 | {report['source']} |",
        f"| 期数 (n_periods) | {report['n_periods']} |",
        f"| 运行时间 | {report['run_at']} |",
        f"| 总体 α | {report['alpha']} |",
        f"| α 拆分 (Sidak) | gate={report['alpha_split']['gate']}, sub={report['alpha_split']['sub']} |",
        f"| 拆分说明 | {report['alpha_split']['note']} |",
        "",
        f"## 判定: **{report['verdict']}**",
        "",
        report["conclusion"],
        "",
    ]
    g1, g2, g3 = report["gate1"], report["gate2"], report["gate3"]
    if g1 is None:
        lines += ["## 门1 无编码对照", "", "样本量不足, 未执行检验。", ""]
    else:
        c, a = g1["chi2"], g1["autocorr"]
        lines += [
            "## 门1 无编码对照",
            "",
            "| 子检验 | 统计量 | p 值 | 显著 (α_sub) |",
            "|---|---|---|---|",
            f"| 卡方均匀 (df={c['df']}) | {_fmt(c['stat'])} | {_fmt(c['p_value'])} | {c['significant']} |",
            f"| 复圆自相关 max\\|z\\| (lag {a['max_z_lag']}, 临界 {_fmt(a['critical_z'])}) | {_fmt(a['max_z'])} | - | {a['significant']} |",
            f"| **门1 通过** | {'是' if g1['passed'] else '否'} | | |",
            "",
        ]
    if g2 is None:
        lines += ["## 门2 复谱主检", "", "样本量不足, 未执行检验。", ""]
    else:
        f = g2["fisher_g"]
        lines += [
            "## 门2 复谱主检",
            "",
            "| 指标 | 值 |",
            "|---|---|",
            f"| Fisher's g | {_fmt(f['g'])} |",
            f"| m (非 DC bin 数) | {f['m']} |",
            f"| p 值 | {_fmt(f['p_value'])} |",
            f"| 峰 bin / 频率 | {_fmt(f['peak_bin'])} / {_fmt(f['peak_freq'])} |",
            f"| 峰相位 (deg) | {_fmt(f['peak_phase_deg'])} |",
            f"| 显著 (α_gate) | {f['significant']} |",
            f"| **门2 通过** | {'是' if g2['passed'] else '否'} |",
            "",
            "| Welch 窗 | 窗数 | 峰 bin | 峰频率 | 峰/中位比 | 峰占比 |",
            "|---|---|---|---|---|---|",
        ]
        for w in g2["welch"]:
            lines.append(f"| {w['window']} | {w['n_windows']} | {w['peak_bin']} | "
                         f"{_fmt(w['peak_freq'])} | {_fmt(w['peak_ratio_median'])} | "
                         f"{_fmt(w['peak_fraction'])} |")
        lines += [
            "",
            "### 峰位一致性",
            "",
            f"| 原始谱/Welch 峰位一致 | {g2['peak_bin_agrees']} |",
            f"| 峰位不稳定 (unstable_peak) | {g2['unstable_peak']} |",
            "",
        ]
    if g3 is None:
        lines += ["## 门3 one-hot 复核", "", "样本量不足, 未执行复核。", ""]
    else:
        lines += [
            "## 门3 one-hot 复核",
            "",
            "| 指标 | 值 |",
            "|---|---|",
            f"| 相位反推号码 (x*) | {_fmt(g3['implicated_number'])} |",
            f"| one-hot g / p | {_fmt(g3['g'])} / {_fmt(g3['p_value'])} |",
            f"| one-hot 峰 bin | {_fmt(g3['peak_bin'])} |",
            f"| 同频复现 (confirmed) | {g3['confirmed']} |",
            f"| 备注 | {g3['note'] if g3['note'] else '门2 未触发'} |",
            "",
        ]
    inv = report["invariance"]
    if inv is None:
        lines += ["## 编码不变性", "", "样本量不足, 未执行。", ""]
    else:
        lines += [
            "## 编码不变性",
            "",
            "| 变换 | Δg | 稳定 |",
            "|---|---|---|",
            f"| 旋转 (c∈{{1,3,8}}, max Δg) | {_fmt(inv['rotation']['g_delta_max'])} | {inv['rotation']['stable']} |",
            f"| 反射 (x→17-x) | {_fmt(inv['reflection']['g_delta'])} | {inv['reflection']['stable']} |",
            f"| 重排 (指纹, 不承诺) | {_fmt(inv['permutation']['g_shift'])} | {inv['permutation']['stable']} |",
            "",
        ]
    rl = report["rolling"]
    if rl is None:
        lines += ["## 滑动窗诊断 (rolling)", "", "样本量不足, 未执行。", ""]
    else:
        lines += [
            "## 滑动窗诊断 (rolling)",
            "",
            "| 指标 | 值 |",
            "|---|---|",
            f"| 窗长/步进 | {rl['window']} / {rl['step']} |",
            f"| 窗口数 | {rl['n_windows']} |",
            f"| min p | {_fmt(rl['min_p'])} |",
            f"| p < α_gate 占比 | {_fmt(rl['frac_below_gate_alpha'])} |",
            f"| 备注 | {rl['note']} |",
            "",
        ]
    lines += ["## 附注"] + [f"- {n}" for n in report["notes"]] + [""]
    return "\n".join(lines)


def render_md_spectral_red(report: Dict) -> str:
    """红球频谱/结构随机性探针报告 -> Markdown: 三路径聚合表 + 极端条目 + conclusion。

    报告 schema 见架构文档 spectral_red_probe; INSUFFICIENT_DATA 时三路径为
    None 显示"样本量不足"。MD 仅聚合+极端（JSON 全量）。

    Args:
        report: red_spectral_report_dict 产出的报告 dict。

    Returns:
        Markdown 文本。
    """
    lines = [
        "# 红球频谱/结构随机性检验报告（spectral_red_probe）",
        "",
        "| 属性 | 值 |",
        "|---|---|",
        f"| 数据源 | {report['source']} |",
        f"| 期数 (n_periods) | {report['n_periods']} |",
        f"| 运行时间 | {report['run_at']} |",
        f"| 总体 α | {report['alpha']} |",
        f"| α 拆分 (Sidak) | path={_fmt(report['alpha_split']['path'])}, "
        f"comp={_fmt(report['alpha_split']['comp'])} |",
        f"| 拆分说明 | {report['alpha_split']['note']} |",
        "",
        f"## 判定: **{report['verdict']}**",
        "",
        report["conclusion"],
        "",
    ]
    if report["path1"] is None:
        lines += ["## 路径1 时间维度（指示序列）", "",
                  "样本量不足, 未执行检验。", "",
                  "## 路径2 横截面（同现矩阵）", "",
                  "样本量不足, 未执行检验。", "",
                  "## 路径3 派生标量", "",
                  "样本量不足, 未执行检验。", ""]
    else:
        p1, p2, p3 = report["path1"], report["path2"], report["path3"]
        c, r, pn = p1["pooled_chi2"], p1["repeat_rate"], p1["per_number"]
        lines += [
            "## 路径1 时间维度（指示序列）",
            "",
            "| 检验 | 统计量 | p 值 | 显著 (α_comp) |",
            "|---|---|---|---|",
            f"| 合并卡方 (df={c['df']}) | {_fmt(c['stat'])} | {_fmt(c['p_value'])} | "
            f"{c['significant']} |",
            f"| 重号率 观测/期望 | {_fmt(r['observed'])} / {_fmt(r['expected'])} | "
            f"z={_fmt(r['z'])} | {r['significant']} |",
            f"| 逐号谱峰显著数 | {pn['fisher_g_significant_count']}/33 | "
            f"min p={_fmt(pn['fisher_g_min_p'])} | - |",
            f"| lag-1 自相关 max|z| (诊断) | {_fmt(pn['lag1_max_z'])} | - | - |",
            "",
            f"边界 p 提示: {c['boundary_note']}（当前合并卡方 p={_fmt(c['p_value'])}）。",
            "",
            "## 路径2 横截面（同现矩阵）",
            "",
            f"矩阵性质: 对称={p2['matrix']['symmetric']}, 对角0={p2['matrix']['diagonal_zero']}, "
            f"总和={p2['matrix']['total']} (期望 30N), 每对期望 {_fmt(p2['matrix']['expected_per_pair'])}, "
            f"观测范围 {p2['matrix']['obs_range']}",
            "",
            "| 检验 | 结果 |",
            "|---|---|",
            f"| FDR 上尾(正向同现) 检出 | {p2['pair_tests']['fdr_sig_positive']}/528 |",
            f"| FDR 下尾(互斥) 检出 | {p2['pair_tests']['fdr_sig_negative']}/528 |",
            f"| max z / min p (上尾) | {_fmt(p2['pair_tests']['max_z'])} / "
            f"{_fmt(p2['pair_tests']['min_p_upper'])} |",
            "",
            "| 子类 | 观测 | 期望 | z | 显著 (临界 3.254) |",
            "|---|---|---|---|---|",
        ]
        for s in p2["subclasses"]:
            lines.append(f"| {s['name']} | {s['observed']} | {_fmt(s['expected'])} | "
                         f"{_fmt(s['z'])} | {s['significant']} |")
        pmi = "; ".join(f"({a},{b}) {_fmt(v)} count={cnt}"
                        for v, a, b, cnt in p2["pmi_top"][:5])
        lines += ["",
                  f"PMI top5 (仅排序展示): {pmi}",
                  "",
                  "## 路径3 派生标量",
                  "",
                  "| 标量 | 观测均值 | null 均值 | z | p | 显著 (α_comp) |",
                  "|---|---|---|---|---|---|",
                  f"| 和值 | {_fmt(p3['sum']['obs_mean'])} | {_fmt(p3['sum']['null_mean'])} | "
                  f"{_fmt(p3['sum']['z'])} | {_fmt(p3['sum']['p_two_sided'])} | "
                  f"{p3['sum']['significant']} |",
                  f"| 跨度 | {_fmt(p3['span']['obs_mean'])} | {_fmt(p3['span']['null_mean'])} | "
                  f"{_fmt(p3['span']['z'])} | - | {p3['span']['significant']} |",
                  f"| 奇偶(奇数个数) | {_fmt(p3['odd_even']['obs_mean_odd'])} | "
                  f"{_fmt(p3['odd_even']['null_mean_odd'])} | "
                  f"chi2={_fmt(p3['odd_even']['chi2'])} (df={p3['odd_even']['df']}, "
                  f"p={_fmt(p3['odd_even']['p'])}) | - | {p3['odd_even']['significant']} |",
                  "",
                  "## 最接近阈值",
                  "",
        ]
        if report["near_miss"]:
            for nm in report["near_miss"]:
                if nm["type"] == "subclass":
                    lines.append(f"- {nm['name']} z={_fmt(nm['z'])} vs 临界 "
                                 f"{nm['critical']}（{nm['note']}）")
                elif nm["type"] == "pooled_chi2":
                    lines.append(f"- 合并卡方 p={_fmt(nm['p_value'])}（{nm['note']}）")
                else:
                    lines.append(f"- {nm['label']} z={_fmt(nm['z'])}（{nm['note']}）")
        else:
            lines.append("- 无")
        lines += ["", "## 附注"] + [f"- {n}" for n in report["notes"]] + [""]
    return "\n".join(lines)


def _fmt_ci(ci) -> str:
    if ci is None or ci[0] is None or ci[1] is None:
        return "-"
    return f"[{ci[0]:.4f}, {ci[1]:.4f}]"


def _suggestion(report: Dict) -> str:
    feat = report["features"][0] if report["features"] else report["strategy"]
    if report["verdict_overall"] == "keep":
        return f"红球平均命中显著高于随机基线, **保留 {feat} 特征开关**"
    return (f"红球平均命中未显著高于随机基线, **回退=关闭 {feat} 特征开关**"
            f"（依据: mean_hits={_fmt(report['red']['mean_hits'])}, "
            f"baseline={_fmt(report['red']['baseline_mean'])}）")


# ================= 特征增量验证(compare_features) =================
FEATURE_STRATEGY_NAMES = ["entropy", "hot_cold", "ac", "crf", "diversity"]


def compare_features(df: pd.DataFrame, features: List[str], horizon: int, train_min: int,
                     n_trials: int, seed: int, pool_size: int,
                     imported_comparison: Optional[List[Dict]] = None) -> Dict:
    """5 特征 × 红/蓝 全量入报告, 不挑拣; 默认关闭的开关由 CLI 显式请求临时启用。"""
    ctx = EvalContext(config={"seed": seed, "pool_size": pool_size})
    strategies_report: List[Dict] = []
    for feat in features:
        strat = f"{feat}" if feat in STRATEGIES else None
        if strat is None:
            strategies_report.append({
                "strategy": feat, "kind": "feature", "features": [feat], "n_periods": 0,
                "train_min": train_min, "run_at": _now(),
                "red": {"mean_hits": None, "baseline_mean": None, "baseline_ci95": [None, None],
                        "delta": None, "delta_ci95": [None, None], "t_stat": None,
                        "t_pvalue": None, "n_ge_50": False, "significant": False,
                        "verdict": "rollback"},
                "blue": {"hit_rate": None, "baseline_rate": BLUE_RANDOM_EXPECT, "delta": None,
                         "delta_ci95": [None, None], "t_pvalue": None, "significant": False,
                         "verdict": "rollback"},
                "verdict_overall": "rollback", "per_period": [],
                "notes": [f"未知特征策略: {feat}, 已跳过"],
            })
            continue
        report = build_report(df, strat, horizon, train_min, ctx, n_trials, seed,
                              features=[feat], imported_comparison=imported_comparison)
        strategies_report.append(report)

    return {
        "kind": "feature_validation",
        "run_at": _now(),
        "horizon": horizon,
        "train_min": train_min,
        "n_trials": n_trials,
        "seed": seed,
        "pool_size": pool_size,
        "strategies": strategies_report,   # 5 特征全量, 红/蓝指标齐备, 不挑拣
        "notes": [
            "特征增量验证: 每特征独立 walk-forward 回测(纯 pandas 打分, 无模型重训)",
            "红/蓝指标全部入报告, 不挑拣; 默认关闭的开关(ac/crf/diversity)由 --features 显式请求临时启用",
            "显著性: mean_hits > 基线+1.96SE 且 n>=50; 判定 keep=保留开关 / rollback=回退关闭",
        ],
        **({"imported_comparison": imported_comparison} if imported_comparison else {}),
    }


def render_md_multi(multi: Dict) -> str:
    """特征增量验证 -> 汇总表 + 各特征明细 + 建议。"""
    lines = [
        "# 特征增量验证报告",
        "",
        f"- run_at: {multi['run_at']} | horizon: {multi['horizon']} | train_min: {multi['train_min']} "
        f"| MC n_trials: {multi['n_trials']} | seed: {multi['seed']} | pool_size: {multi['pool_size']}",
        "",
        "## 汇总（5 特征 × 红/蓝 全量）",
        "",
        "| 特征 | 红球平均命中 | 红球基线 | 红球Δ | 红球 t_p | 红球判定 | 蓝球命中率 | 蓝球基线 | 蓝球Δ | 蓝球判定 | 总体判定 |",
        "|---|---|---|---|---|---|---|---|---|---|---|",
    ]
    for s in multi["strategies"]:
        r, b = s["red"], s["blue"]
        lines.append(
            f"| {s['strategy']} | {_fmt(r['mean_hits'])} | {_fmt(r['baseline_mean'])} | "
            f"{_fmt(r['delta'])} | {_fmt(r['t_pvalue'])} | {r['verdict']} | "
            f"{_fmt(b['hit_rate'])} | {_fmt(b['baseline_rate'])} | {_fmt(b['delta'])} | "
            f"{b['verdict']} | {s['verdict_overall']} |"
        )
    lines += ["", "## 建议", ""]
    for s in multi["strategies"]:
        if s["per_period"]:
            lines.append(f"- **{s['strategy']}** ({s['verdict_overall']}): {_suggestion(s)}")
        else:
            lines.append(f"- **{s['strategy']}**: 未执行 ({s['notes'][-1]})")
    lines += ["", "## 附注"] + [f"- {n}" for n in multi["notes"]]
    if "imported_comparison" in multi:
        lines += ["", "## 历史对比(imported: analysis/results/comparison_table.csv)", "",
                  "| model | mode | red_mean_overlap | red_hit_ge3 | blue_top1_acc | n_periods |",
                  "|---|---|---|---|---|---|"]
        for row in multi["imported_comparison"]:
            lines.append(f"| {row.get('model', '-')} | {row.get('mode', '-')} | "
                         f"{row.get('red_mean_overlap', '-')} | {row.get('red_hit_ge3', '-')} | "
                         f"{row.get('blue_top1_acc', '-')} | {row.get('n_periods', '-')} |")
    lines.append("")
    return "\n".join(lines)


# ================= --import-comparison =================
def import_comparison_table(path: Optional[Path] = None) -> Optional[List[Dict]]:
    """读取既有 analysis/results/comparison_table.csv, 汇总进报告(不重复计算)。"""
    p = path or (ROOT / "analysis/results/comparison_table.csv")
    if not p.exists():
        return None
    df = pd.read_csv(p)
    return df.to_dict(orient="records")


# ================= CLI =================
def list_strategies() -> None:
    print(f"{'策略':<12}{'kind':<10}{'特征':<24}{'开关(默认)'}")
    print("-" * 60)
    for name, meta in STRATEGIES.items():
        toggle = FEATURE_TOGGLES.get(name, "-")
        feats = ",".join(meta["features"]) if meta["features"] else "-"
        print(f"{name:<12}{meta['kind']:<10}{feats:<24}{toggle}")


def main(argv: Optional[List[str]] = None) -> Dict:
    ap = argparse.ArgumentParser(description="双色球策略回测标准化 + 特征增量验证")
    ap.add_argument("--strategy", default=None, help="单策略名(如 entropy / freq / model:pg / spectral)")
    ap.add_argument("--features", default=None, help="逗号分隔特征列表(如 ac,entropy,hot_cold,crf,diversity)")
    ap.add_argument("--spectral", action="store_true",
                    help="频谱平坦性独立探针模式(三关检验, 与 --strategy/--features 互斥)")
    ap.add_argument("--spectral-red", action="store_true",
                    help="红球频谱/结构随机性独立探针模式(三路径检验, 与 --spectral/--strategy/--features 互斥)")
    ap.add_argument("--window", type=int, default=64,
                    help="Welch 主窗长(仅 --spectral 用)")
    ap.add_argument("--horizon", type=int, default=EVAL_CONFIG["horizon"])
    ap.add_argument("--out", default=None, help="输出路径, 按扩展名 .json/.md 决定格式")
    ap.add_argument("--smoke", action="store_true", help="horizon=5 链路冒烟")
    ap.add_argument("--list", action="store_true", help="枚举策略注册表")
    ap.add_argument("--import-comparison", action="store_true",
                    help="汇总 analysis/results/comparison_table.csv 既有结果进报告")
    ap.add_argument("--train-min", type=int, default=EVAL_CONFIG["train_min"])
    ap.add_argument("--n-trials", type=int, default=EVAL_CONFIG["n_trials"])
    ap.add_argument("--seed", type=int, default=EVAL_CONFIG["seed"])
    ap.add_argument("--pool-size", type=int, default=EVAL_CONFIG["pool_size"])
    args = ap.parse_args(argv)

    if args.list:
        list_strategies()
        return {"kind": "list"}

    horizon = 5 if args.smoke else args.horizon
    train_min = args.train_min
    n_trials = args.n_trials
    seed = args.seed
    pool_size = args.pool_size

    feats = [f.strip() for f in args.features.split(",")] if args.features else []
    imported = import_comparison_table() if args.import_comparison else None

    df = load_data()
    run_at = _now()

    # 默认输出路径
    default_name = "report_smoke" if args.smoke else "report_default"
    out = Path(args.out) if args.out else DC_SSQ / "analysis/results" / f"{default_name}.json"

    # ---------- 红球频谱探针(独立模式, 分支最前; 与 --spectral/--strategy/--features 互斥) ----------
    if args.spectral_red:
        if args.spectral or args.strategy or args.features:
            ap.error("--spectral-red 与 --spectral/--strategy/--features 互斥")
        out = Path(args.out) if args.out else DC_SSQ / "analysis/results/spectral_red_probe.json"
        reds = df[RED_COLS].to_numpy(dtype=int)
        result = run_red_spectral_test(reds)
        report = red_spectral_report_dict(result, run_at=run_at,
                                          source="ml/data/1.csv (Red1..Red6)")
        write_report(report, out)
        if out.suffix.lower() != ".md":
            write_report(report, out.with_suffix(".md"))
        print(f"[evaluate] 红球频谱探针完成: verdict={report['verdict']} "
              f"n={report['n_periods']} -> {out}")
        return report

    # ---------- 频谱探针(独立模式, 分支最前; 与 --strategy/--features 互斥) ----------
    if args.spectral:
        if args.strategy or args.features:
            ap.error("--spectral 与 --strategy/--features 互斥")
        out = Path(args.out) if args.out else DC_SSQ / "analysis/results/spectral_probe.json"
        blues = df[BLUE_COLS[0]].astype(int).to_numpy()
        result = run_three_gate_test(blues, window=args.window)
        report = spectral_report_dict(result, run_at=run_at, source="ml/data/1.csv (Blue1)")
        write_report(report, out)
        if out.suffix.lower() != ".md":
            write_report(report, out.with_suffix(".md"))
        print(f"[evaluate] 频谱探针完成: verdict={report['verdict']} "
              f"n={report['n_periods']} -> {out}")
        return report

    # ---------- 单策略 ----------
    if args.strategy:
        sname = args.strategy
        if sname not in STRATEGIES:
            print(f"未知策略: {sname}", file=sys.stderr)
            list_strategies()
            sys.exit(2)
        # 特征开关: 默认关闭的特征需 --features 显式请求才启用(评测覆盖)
        if sname in FEATURE_TOGGLES and not FEATURE_TOGGLES[sname] and sname not in feats:
            print(f"特征开关 {sname}=false(默认关闭)。"
                  f"可用 --features {sname} 显式启用本次评测", file=sys.stderr)
            sys.exit(2)
        ctx = EvalContext(config={"seed": seed, "pool_size": pool_size})
        if sname == "model:pg":
            pg = load_pg_probs()
            if pg is None:
                print("警告: PG 概率不可用, model:pg 将 graceful skip 并在报告中标注", file=sys.stderr)
            ctx.probs = pg
            extra = [f"PG 冻结概率: run_at={pg['run_at']} models={pg['models']}"] if pg else []
        elif sname == "spectral":
            extra = [
                "检验器不预测：spectral 策略输出仅用于诊断对照",
                "蓝球=近窗谱峰相位外推（无显著峰则随机）",
            ]
        else:
            extra = []
        report = build_report(df, sname, horizon, train_min, ctx, n_trials, seed,
                              features=feats or None, extra_notes=extra,
                              run_at=run_at, imported_comparison=imported)
        write_report(report, out)
        if out.suffix.lower() != ".md":
            write_report(report, out.with_suffix(".md"))
        print(f"[evaluate] {sname} 回测完成: n_periods={report['n_periods']} "
              f"red_mean={_fmt(report['red']['mean_hits'])} "
              f"verdict={report['verdict_overall']} -> {out}")
        return report

    # ---------- 特征增量验证(compare_features) ----------
    if feats:
        multi = compare_features(df, feats, horizon, train_min, n_trials, seed,
                                 pool_size, imported_comparison=imported)
        if out.suffix.lower() == ".md":
            out.write_text(render_md_multi(multi), encoding="utf-8")
        else:
            out.parent.mkdir(parents=True, exist_ok=True)
            out.write_text(json.dumps(_jsonable(multi), ensure_ascii=False, indent=2),
                           encoding="utf-8")
        print(f"[evaluate] 特征增量验证完成: {len(multi['strategies'])} 个特征全量入报告 -> {out}")
        return multi

    # ---------- 默认: 单策略 entropy(关闭则退回 freq) ----------
    default_strategy = "entropy" if FEATURE_TOGGLES["entropy"] else "freq"
    ctx = EvalContext(config={"seed": seed, "pool_size": pool_size})
    report = build_report(df, default_strategy, horizon, train_min, ctx, n_trials, seed,
                          run_at=run_at, imported_comparison=imported)
    write_report(report, out)
    if out.suffix.lower() != ".md":
        write_report(report, out.with_suffix(".md"))
    print(f"[evaluate] 默认策略 {default_strategy} 回测完成 -> {out}")
    return report


if __name__ == "__main__":
    main()
