#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""从 PostgreSQL 读取最新一次模型预测概率，集成生成 5 组双色球候选号码。

逻辑(完全基于 ml/main.py 已训练模型的输出, 不做额外预测):
  1. 取 PG ssq.model_predictions 中每个模型各自最新 run_at 的概率(每模型每
     球种独立取最新, 不要求同一次 run 共享 run_at), 实际覆盖全部已入库模型
 (当前 6 模型: rf/lightgbm/cnn_reg/lstm/transformer/cdm; 由 batch_predict_pg.MODELS 决定)。
  2. 红球(1-33)/蓝球(1-16) 分别对"该侧有数据的模型"概率取均值 -> 集成概率。
  3. 红球: 在集成概率上做受控随机加权抽样(softmax 温度), 生成5注,
     每注6个互不相同的号, 并约束奇偶比∈{2:4,3:3,4:2}、大小比(1-16小/17-33大)∈{2:4,3:3,4:2}。
  4. 蓝球: 取集成概率 Top 并结合受控随机, 每注1个。
  5. 输出每注的选取依据(命中哪些模型的高概率号)。

可选后处理层(默认开启, 均不提升命中率, 仅概率诚实化/风险分层):
  - James-Stein 收缩(--no-shrink 关闭): 均值集成后向均匀先验收缩, 对抗过拟合噪声;
  - Conformal 候选集(--no-conformal 关闭): 带理论覆盖率保证的风险分层解释。

用法:
  .venv/bin/python select_numbers.py                 # 生成5组并打印
  .venv/bin/python select_numbers.py --groups 5 --seed 42
  .venv/bin/python select_numbers.py --wheel         # 旋转矩阵覆盖模式(红球池 Top18)
  .venv/bin/python select_numbers.py --wheel --pool-size 15 --max-notes 30 --no-popularity
  .venv/bin/python select_numbers.py --no-shrink     # 关闭收缩(与旧行为一致)
"""
from __future__ import annotations
import argparse
import json
import sys
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

import numpy as np
import psycopg
from ml.pg_conn import pg_dict

from ml.config import POPULARITY_CONFIG, WHEEL_CONFIG
from ml.popularity import combo_popularity, sample_with_popularity
from wheel import CoverResult, greedy_cover
# C1 conformal 风险分层(研究简报 2026-08-17 [3], 工程落地 2026-08-19):
# 把集成概率升级为带理论覆盖率保证的候选集合, 仅作可解释风险分层, 不改变命中率。
from ml.conformal.conformal_predict import build_from_history as _conformal_build

PG = pg_dict()  # 凭证从 ~/.hermes/.env 的 DATABASE_URL 读, 不硬编码
SCHEMA = "ssq"
MODELS = ["rf", "lightgbm", "cnn_reg", "lstm", "transformer", "cdm"]  # 仅文档/兼容用途, 生产以 batch_predict_pg.MODELS 为准


def load_latest_probs(conn, method: str = "mean", tau: float = 8000.0):
    """读取最新一次 run 的各模型概率, 返回 (red_mean[33], blue_mean[16], run_at, models)。

    Args:
        conn: psycopg 连接。
        method: 多模型融合方式。'mean'=等权均值(默认, 向后兼容);
                'ebma'=按历史开奖对数似然 softmax 加权(详见 ml.ensemble)。
        tau: EBMA 温度(仅 method='ebma' 时生效)。默认 8000.0。

        重要诚实声明: 双色球为独立均匀随机过程, 各模型的"历史表现差异"
        主要是抽样噪声的累积放大(实测红球 6 模型 3489 期累计 log-likelihood
        极差约 7900 nat), 并非真预测技能。因此:
          - tau 过小(如 50)会让 softmax 把噪声当信号, 权重坍缩到单一模型;
          - tau 调大到 ~8000 才使权重接近等权(这是更诚实的默认, 不假装某模型更优);
          - EBMA 模式本质是"模型差异诊断探针", 不代表集成预测更准。
        数学上任何融合都无法提升随机过程的命中率。
    """
    from ml.ensemble import integrate_redblue  # 延迟导入避免循环依赖

    with conn.cursor() as cur:
        # 每模型每球种独立取最新 run_at(避免"必须一次跑完所有模型共享 run_at"的耦合)
        cur.execute(f"""
            SELECT model, ball_type, num, prob FROM {SCHEMA}.model_predictions
            WHERE (model, ball_type, run_at) IN (
                SELECT model, ball_type, MAX(run_at)
                FROM {SCHEMA}.model_predictions GROUP BY model, ball_type
            )
            ORDER BY model, ball_type, num
        """)
        red = {}
        blue = {}
        for model, btype, num, prob in cur.fetchall():
            red.setdefault(model, np.zeros(33))
            blue.setdefault(model, np.zeros(16))
            if btype == "red":
                red[model][num - 1] = prob
            elif btype == "blue":
                blue[model][num - 1] = prob
    # run_at 不再单一(各模型独立), 用各模型最新 run_at 的集合表示
    run_at = "per-model-latest"
    models = list(red.keys())
    # 仅保留该侧有数据的模型
    red_models = {m: red[m] for m in models if red[m].sum() > 0}
    blue_models = {m: blue[m] for m in models if blue[m].sum() > 0}
    if not red_models or not blue_models:
        raise RuntimeError(f"集成失败: 红球模型={list(red_models)} 蓝球模型={list(blue_models)}")

    if method == "mean":
        red_mean = np.stack(list(red_models.values())).mean(axis=0)
        blue_mean = np.stack(list(blue_models.values())).mean(axis=0)
    elif method == "ebma":
        red_mean, blue_mean, _rw, _bw = integrate_redblue(
            red_models, blue_models, method="ebma", tau=tau)
    else:
        raise ValueError(f"未知 method={method}")
    return red_mean, blue_mean, run_at, models


def _sample_red(red_prob, rng, temperature=0.6, popularity_fn=None, lambda_=0.3):
    """按受控随机加权抽样6个不重复红球, 约束奇偶/大小比。

    popularity_fn 不为 None 时: 先按温度采样生成候选注, 再按流行度惩罚加权
    重采样(偏向冷门组合); 为 None 走原逻辑(向后兼容, generate 默认路径不变)。
    """
    if popularity_fn is not None:
        return sample_with_popularity(red_prob, rng, temperature=temperature,
                                      lambda_=lambda_, popularity_fn=popularity_fn)
    for _ in range(200):
        p = np.power(red_prob, 1.0 / temperature)
        p = p / p.sum()
        picks = rng.choice(33, size=6, replace=False, p=p) + 1
        odds = sum(1 for x in picks if x % 2 == 1)
        big = sum(1 for x in picks if x > 16)
        if odds in (2, 3, 4) and big in (2, 3, 4):
            return sorted(picks.tolist())
    # 兜底: 直接取Top6
    return sorted((np.argsort(red_prob)[-6:] + 1).tolist())


def _sample_blue(blue_prob, rng, temperature=0.7):
    p = np.power(blue_prob, 1.0 / temperature)
    p = p / p.sum()
    return int(rng.choice(16, size=1, p=p)[0] + 1)


def generate(red_mean, blue_mean, groups=5, seed=42):
    rng = np.random.default_rng(seed)
    out = []
    for g in range(groups):
        reds = _sample_red(red_mean, rng)
        blue = _sample_blue(blue_mean, rng)
        # 选取依据: 该红球在集成概率中的排名(前8为热号)
        red_rank = {int(n) + 1: int(r) + 1 for r, n in enumerate(np.argsort(red_mean)[::-1])}
        blue_rank = {int(n) + 1: int(r) + 1 for r, n in enumerate(np.argsort(blue_mean)[::-1])}
        hot_reds = [n for n in reds if red_rank[n] <= 8]
        out.append({
            "group": g + 1,
            "red": reds,
            "blue": blue,
            "hot_reds": hot_reds,
            "blue_rank": blue_rank[blue],
        })
    return out


def build_conformal(conn, alpha: float = 0.90, min_pairs: int = 8):
    """从 PG 历史预测批次校准 conformal 集合(风险分层, 非预测增益)。

    对齐逻辑: batch_predict_pg 在 data_date 当天预测的是该日期之后的**下一期**
    开奖号(预测先于开奖), 故每批概率应 vs 该 data_date 之后最早一期开奖。
    开奖号取 **1.csv**(生产数据源, 永远最新) 而非 draw_history(update_ssq 已自动 upsert 同步,
    但 1.csv 仍是权威源), 与管线"生产路径读 1.csv"约定一致。
    每批概率取等权集成(与 load_latest_probs 同口径), 对齐到下一期真实开奖号,
    调 _conformal_build 校准红/蓝 conformity 阈值。

    返回 {"red": ConformalSet, "blue": ConformalSet, "n_pairs": int} 或 None(样本不足)。
    诚实声明: 覆盖率保证的是\"集合包含开奖号\"的概率, 不提升命中率(不可能)。
    """
    # 1) 拉全部预测批次(按 data_date 升序)
    with conn.cursor() as cur:
        cur.execute(f"""
            SELECT run_at, data_date, model, ball_type, num, prob
            FROM {SCHEMA}.model_predictions
            ORDER BY data_date, run_at, model, ball_type, num
        """)
        rows = cur.fetchall()
    if not rows:
        return None

    # 2) 开奖号取 1.csv(生产数据源, 最新) —— draw_history 已由 update_ssq 自动同步, 但 1.csv 为权威源
    import csv as _csv
    from pathlib import Path
    csv_path = Path(__file__).resolve().parent / "ml/data/1.csv"
    draw_by_date = {}
    if csv_path.exists():
        with csv_path.open("r", encoding="utf-8-sig", newline="") as f:
            for r in _csv.DictReader(f):
                try:
                    ddate = datetime.strptime(r["dDate"], "%Y-%m-%d").date()
                except (KeyError, ValueError):
                    continue
                reds = [int(r[f"Red{i}"]) for i in range(1, 7)]
                blue = int(r["Blue1"])
                draw_by_date[ddate] = (reds, blue)
    if not draw_by_date:
        return None

    # 组织批次: {(data_date, run_at): {model: {red:33, blue:16}}}
    from collections import defaultdict
    batches = defaultdict(lambda: defaultdict(lambda: {"red": np.zeros(33),
                                                        "blue": np.zeros(16)}))
    for run_at, data_date, model, btype, num, prob in rows:
        key = (data_date, run_at)
        if btype == "red":
            batches[key][model]["red"][num - 1] = prob
        else:
            batches[key][model]["blue"][num - 1] = prob

    # 按 data_date 升序遍历批次, 对齐下期开奖
    red_hist, blue_hist, red_draws, blue_draws = [], [], [], []
    for (ddate, _rt) in sorted(batches.keys()):
        # 下一期 = 比 ddate 晚的最早一期
        nxt = None
        for dd in draw_by_date:
            if dd > ddate:
                nxt = dd
                break
        if nxt is None:
            continue
        reds6, blue1 = draw_by_date[nxt]
        # 等权集成该批有数据的模型
        bm = batches[(ddate, _rt)]
        red_models = [m for m in bm if bm[m]["red"].sum() > 0]
        blue_models = [m for m in bm if bm[m]["blue"].sum() > 0]
        if not red_models or not blue_models:
            continue
        red_prob = np.stack([bm[m]["red"] for m in red_models]).mean(axis=0)
        blue_prob = np.stack([bm[m]["blue"] for m in blue_models]).mean(axis=0)
        red_hist.append(red_prob)
        blue_hist.append(blue_prob)
        red_draws.append(reds6)
        blue_draws.append(blue1)

    if len(red_draws) < min_pairs:
        return None

    cs = _conformal_build(red_hist, blue_hist, red_draws, blue_draws, alpha=alpha)
    cs["n_pairs"] = len(red_draws)
    return cs


def apply_conformal(cs, red_mean, blue_mean):
    """给定当前集成概率, 返回 conformal 候选集(风险分层, 不改选号)。"""
    if cs is None:
        return None
    red_set = cs["red"].predict_set(red_mean)
    blue_set = cs["blue"].predict_set(blue_mean)
    return {
        "red_set": red_set,
        "blue_set": blue_set,
        "red_summary": _conformal_summarize(cs["red"], red_mean),
        "blue_summary": _conformal_summarize(cs["blue"], blue_mean),
        "n_pairs": cs.get("n_pairs", 0),
    }


def _conformal_summarize(cs, prob):
    from ml.conformal.conformal_predict import summarize_coverage
    return summarize_coverage(cs, prob)


def _popularity_off(reds):
    """--no-popularity 时使用的流行度函数: 不计算流行度(返回 None)。"""
    return None


def build_wheel_tickets(red_mean, blue_mean, pool_size=18, max_notes=30, restarts=3,
                        seed=42, popularity_fn=None, lambda_=0.3):
    """旋转矩阵覆盖模式: 红球 Top-pool_size 概率池 -> 贪心覆盖 -> 每注配 1 个蓝球。

    Args:
        red_mean: 33 维红球集成概率。
        blue_mean: 16 维蓝球集成概率。
        pool_size: 概率 Top-N 红球池大小(6..33)。
        max_notes / restarts / seed: 透传 wheel.greedy_cover。
        popularity_fn: 每注流行度计算函数(reds -> float); None 时用默认
            combo_popularity(规则版)。
        lambda_: 流行度惩罚系数, 保留给惩罚加权路径(与 sample_with_popularity
            同款参数); 默认路径不影响 combo_popularity 输出。

    Returns:
        {"tickets": [{"reds": [...], "blue": n, "popularity": float|None}, ...],
         "coverage": {CoverResult 字段 + pool/pool_size}}
    """
    red_mean = np.asarray(red_mean, dtype=float)
    blue_mean = np.asarray(blue_mean, dtype=float)
    if len(red_mean) != 33 or len(blue_mean) != 16:
        raise ValueError(f"red_mean 长度必须为33, blue_mean 必须为16, "
                         f"实际 {len(red_mean)}/{len(blue_mean)}")
    if not (6 <= pool_size <= 33):
        raise ValueError(f"pool_size 必须在 6-33 之间, 实际 {pool_size}")
    top = np.argsort(red_mean)[::-1][:pool_size]
    pool = sorted((top + 1).tolist())
    res: CoverResult = greedy_cover(pool, k=6, t=4, max_notes=max_notes,
                                    restarts=restarts, seed=seed)
    blue_rng = np.random.default_rng(seed + 1)
    tickets = []
    for reds in res.tickets:
        blue = _sample_blue(blue_mean, blue_rng, temperature=0.7)
        popularity = popularity_fn(reds) if popularity_fn is not None \
            else combo_popularity(reds)
        tickets.append({"reds": reds, "blue": blue, "popularity": popularity})
    coverage = {
        "n_notes": res.n_notes,
        "covered_4subsets": res.covered_4subsets,
        "total_4subsets": res.total_4subsets,
        "four_subset_coverage": res.four_subset_coverage,
        "pass_rate": res.pass_rate,
        "pass_rate_sampled": res.pass_rate_sampled,
        "converged": res.converged,
        "max_notes": res.max_notes,
        "pool": pool,
        "pool_size": len(pool),
    }
    return {"tickets": tickets, "coverage": coverage}


def _run_wheel(red_mean, blue_mean, args, run_at):
    """--wheel 模式: 生成并打印覆盖注单 + 覆盖率报告。"""
    pop_fn = _popularity_off if args.no_popularity else None
    result = build_wheel_tickets(red_mean, blue_mean, pool_size=args.pool_size,
                                 max_notes=args.max_notes, restarts=3, seed=args.seed,
                                 popularity_fn=pop_fn, lambda_=args.popularity_lambda)
    cov = result["coverage"]
    if args.json:
        print(json.dumps({"run_at": str(run_at), "mode": "wheel", **result},
                         ensure_ascii=False, indent=2))
        return
    print(f"模型集成预测 ({run_at}) | 旋转矩阵覆盖模式: 红球池 Top-{cov['pool_size']}")
    print("=" * 56)
    for i, t in enumerate(result["tickets"], 1):
        pop = t["popularity"]
        pop_s = f"{pop:.3f}" if isinstance(pop, float) else "—"
        print(f"注{i:02d}: 红球 {t['reds']} + 蓝球 {t['blue']:02d} | 流行度 {pop_s}")
    print("=" * 56)
    pr_s = f"{cov['pass_rate'] * 100:.2f}%"
    if cov["pass_rate_sampled"] is not None:
        pr_s += " (抽样±0.35%)"
    else:
        pr_s += " (精确)"
    print(f"[覆盖报告] 4-子集覆盖率: {cov['covered_4subsets']}/{cov['total_4subsets']} "
          f"= {cov['four_subset_coverage'] * 100:.2f}%")
    print(f"[覆盖报告] 6-子集通过率: {pr_s} | 注数 {cov['n_notes']}/{cov['max_notes']} "
          f"| 收敛: {cov['converged']} | 池大小: {cov['pool_size']}")
    print("注: 覆盖设计为概率性保证(6-子集通过率≥99% 阈值在池≤16/25注可达), 仅供娱乐参考")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--groups", type=int, default=5)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--json", action="store_true", help="输出 JSON")
    ap.add_argument("--wheel", action="store_true",
                    help="旋转矩阵覆盖模式(概率性保证, 覆盖 Top-N 概率池)")
    ap.add_argument("--pool-size", type=int, default=WHEEL_CONFIG.get("pool_size", 18),
                    help="红球概率池大小(6..33)")
    ap.add_argument("--max-notes", type=int, default=WHEEL_CONFIG.get("max_notes", 30),
                    help="旋转矩阵最大注数")
    ap.add_argument("--no-popularity", action="store_true", help="关闭流行度惩罚")
    ap.add_argument("--popularity-lambda", type=float,
                    default=POPULARITY_CONFIG.get("lambda", 0.3),
                    help="流行度惩罚系数 λ")
    ap.add_argument("--ensemble", choices=["mean", "ebma"], default="mean",
                    help="多模型融合方式: mean=等权(默认) | ebma=历史对数似然加权")
    ap.add_argument("--ensemble-tau", type=float, default=8000.0,
                    help="EBMA 温度(仅 --ensemble ebma 生效): 随机过程上模型差异是噪声,"
                         "tau~8000 接近等权, 过小则坍缩到单一模型")
    ap.add_argument("--no-conformal", action="store_true",
                    help="关闭 conformal 风险分层候选集输出(默认开启, 仅解释层不改选号)")
    ap.add_argument("--conformal-alpha", type=float, default=0.90,
                    help="conformal 目标覆盖率 1-α (默认 0.90)")
    ap.add_argument("--no-shrink", action="store_true",
                    help="关闭 James-Stein 收缩(默认开启: 均值集成后向均匀先验收缩,"
                         "对抗模型过拟合噪声; 不提升命中率, 概率更诚实)")
    ap.add_argument("--shrink-alpha", type=float, default=1.0,
                    help="James-Stein 收缩强度系数 ∈[0,1]; 1=标准(默认), 0=不收缩")
    args = ap.parse_args()

    conn = psycopg.connect(**PG)
    try:
        red_mean, blue_mean, run_at, _ = load_latest_probs(
            conn, method=args.ensemble, tau=args.ensemble_tau)
    finally:
        conn.close()

    # P1 James-Stein 收缩后处理(研究简报 2026-08-22 [1], 落地 2026-08-22):
    # 均值集成后向均匀先验收缩, 本质是对模型在随机数据上学到的偏离均匀的
    # 噪声做正则化。与 EBMA(模型权重融合)正交。预期命中率无显著变化(FLAT),
    # 价值在概率诚实化与可辩护性。
    if not args.no_shrink:
        from ml.shrinkage import shrink_red_blue
        red_mean, blue_mean = shrink_red_blue(red_mean, blue_mean,
                                              sigma2=1.0, alpha=args.shrink_alpha)

    if args.wheel:
        _run_wheel(red_mean, blue_mean, args, run_at)
        return

    groups = generate(red_mean, blue_mean, args.groups, args.seed)

    # C1 conformal 风险分层(仅解释层, 不影响选号)
    conf = None
    if not args.no_conformal:
        conn2 = psycopg.connect(**PG)
        try:
            conf = build_conformal(conn2, alpha=args.conformal_alpha)
        finally:
            conn2.close()

    if args.json:
        out = {"run_at": str(run_at), "groups": groups}
        if conf is not None:
            c = apply_conformal(conf, red_mean, blue_mean)
            out["conformal"] = c
        print(json.dumps(out, ensure_ascii=False, indent=2))
        return

    print(f"模型集成预测 ({run_at}) | 红球Top8: "
          f"{[int(x)+1 for x in np.argsort(red_mean)[::-1][:8]]} "
          f"蓝球Top5: {[int(x)+1 for x in np.argsort(blue_mean)[::-1][:5]]}")
    print("=" * 56)
    for g in groups:
        print(f"第{g['group']}组: 红球 {g['red']} + 蓝球 {g['blue']:02d}")
        print(f"       热号(集成概率前8): 红球 {g['hot_reds']} | 蓝球排名 #{g['blue_rank']}")
    print("=" * 56)
    if conf is not None:
        c = apply_conformal(conf, red_mean, blue_mean)
        rs, bs = c["red_summary"], c["blue_summary"]
        print(f"[Conformal 风险分层] 校准样本={c['n_pairs']}期, α={args.conformal_alpha}")
        print(f"  红球候选集(大小={len(c['red_set'])}): {c['red_set']}")
        print(f"  蓝球候选集(大小={len(c['blue_set'])}): {c['blue_set']}")
        print(f"  红球: {rs['coverage_claim']} | 蓝球: {bs['coverage_claim']}")
        if c['n_pairs'] < 30:
            print(f"  注: 校准样本仅 {c['n_pairs']} 期(边际覆盖率保证仍成立, 但经验集合大小估计有噪声; "
                  f"月度重训积累批次后更稳)")
        print(f"  注: conformal 集合为理论覆盖率保证(不提升命中率, 仅可解释风险分层)")
    print("注: 以上基于 ML 模型历史概率集成, 仅供娱乐参考, 不保证中奖。")


if __name__ == "__main__":
    main()
