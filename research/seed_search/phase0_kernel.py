#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""SSQ 种子搜索 Phase 0 内核: 号码频率 + 种子→注单→得分。

核心数学恒等式(证明见 SEED_SEARCH_PLAN §12):
  固定 30 注对 N 期开奖的 red_mean_hits
  = (1/30) Σ_{注} Σ_{n∈注} freq_red(n)/N
  = mean(p_red[180 个红球号码])
即"选号→对齐"的命中均值,精确等价于"所选号码的历史频率均值"。
这一恒等式把 O(30注 × N期) 的逐期对齐,降为 O(30次抽样) 的查表求和,
是 Phase 0 能跑到百万级种子的关键。

口径 A(本模块默认): 种子生成固定 30 注, 对窗口内全部开奖期求命中均值。
候选池 = 全域均匀(33 球无放回), 基线可解析 = 6×6/33 = 1.0909。
"""
from __future__ import annotations

import sys
from pathlib import Path

PROJ = Path("/home/hermes/workspace/python/SSQ")
sys.path.insert(0, str(PROJ))

import numpy as np  # noqa: E402


def load_draws(conn) -> tuple[np.ndarray, np.ndarray]:
    """从 PG 读全量开奖, 返回 reds (N,6) 0-indexed, blues (N,) 0-indexed。"""
    with conn.cursor() as cur:
        cur.execute(
            """SELECT red1, red2, red3, red4, red5, red6, blue1
               FROM ssq.draw_history ORDER BY dnum"""
        )
        rows = cur.fetchall()
    arr = np.asarray(rows, dtype=np.int64)
    reds = arr[:, :6] - 1   # 1-indexed → 0-indexed
    blues = arr[:, 6] - 1
    return reds, blues


def build_freq(reds: np.ndarray, blues: np.ndarray,
               window: int | None = None) -> tuple[np.ndarray, np.ndarray, int]:
    """按最近 window 期(默认全量)统计红/蓝频率 → p_red(33,), p_blue(16,), N。"""
    if window is not None:
        reds = reds[-window:]
        blues = blues[-window:]
    n = reds.shape[0]
    p_red = np.bincount(reds.ravel(), minlength=33).astype(np.float64) / n
    p_blue = np.bincount(blues.ravel(), minlength=16).astype(np.float64) / n
    return p_red, p_blue, n


def seed_to_score(seed: int, p_red: np.ndarray, p_blue: np.ndarray,
                  n_notes: int = 30) -> tuple[float, float]:
    """种子 → 固定 30 注(全域均匀) → (red_mean_hits, blue_hit_rate)。

    红球: 每注从 33 球无放回抽 6 个; 蓝球: 每注从 16 球抽 1 个。
    得分 = 所选号码的历史频率均值(恒等式见模块 docstring)。
    """
    rng = np.random.default_rng(seed)
    reds = np.stack([rng.choice(33, size=6, replace=False)
                     for _ in range(n_notes)])          # (30,6) 0-indexed
    blues = rng.integers(0, 16, size=n_notes)           # (30,) 0-indexed
    # red_mean_hits = (1/n_notes) Σ_注 Σ_{n∈注} p_red[n]
    #              = p_red[reds].sum() / n_notes  (每注 6 红各贡献一次命中)
    red_score = float(p_red[reds].sum() / n_notes)
    blue_score = float(p_blue[blues].mean())            # 每注 1 蓝, 命中 = p_blue[blue]
    return red_score, blue_score


def seed_to_tickets(seed: int, n_notes: int = 30) -> tuple[np.ndarray, np.ndarray]:
    """种子 → 注单号码(仅用于 0.5 相关种子独立性检查)。返回 reds(30,6), blues(30,) 0-indexed。"""
    rng = np.random.default_rng(seed)
    reds = np.stack([rng.choice(33, size=6, replace=False)
                     for _ in range(n_notes)])
    blues = rng.integers(0, 16, size=n_notes)
    return reds, blues


# 理论基线(全域均匀, 任意窗口): 一注 6 红命中期望 = 6×6/33
RED_BASE = 6 * 6 / 33
BLUE_BASE = 1 / 16
