#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""双色球旋转矩阵: 加权贪心覆盖生成器 (标准库-only, 无第三方依赖)。

目标: 从红球池中挑选至多 max_notes 注(每注 6 个红球), 使"至少覆盖 1 个
4-子集"的 6-子集(开奖结果形态)数量最大化 —— 即最大化 6-子集通过率 pass_rate
(6-子集通过 = 其 15 个 4-子集中任一被某注覆盖, 对应开奖 6 红命中某注 ≥4 红)。

算法 (加权贪心集合覆盖, 位掩码编码):
  1. 预计算: 全部 C(N,6) 个 6-子集 S 及其 15 个 4-子集; 反向索引 q -> [含 q 的 S];
     候选注 = 全部 C(N,6) 个组合, 每候选预存其 15 个 4-子集位掩码。
  2. 初始化: covered=∅; act_count[S]=0; weight[q] = 含 q 的未激活 6-子集数。
  3. 每轮: score(候选) = Σ weight[q] (q∈候选 且 q∉covered); 取 score 最大者
     (并列取候选序第一个, 候选序按 seed 打乱保证确定性); score=0 -> 收敛提前终止。
  4. 更新: 对候选的每个新覆盖 4-子集 q: covered.add(q); 对每个 S∋q:
     act_count[S]+=1, 若恰变 1(新激活) -> 对 S 其余 14 个未覆盖 4-子集 q2: weight[q2]-=1。
  5. restarts 次(默认 3): 不同 seed 打乱候选序各跑一遍, 按 pass_rate 取最优。
  6. pass_rate: N≤20 精确穷举(C(20,6)=38760 可枚举); N>20 抽样估计
     (sampling_n=20000, 固定 seed, ±0.35% 抽样误差)。

用法:
  from wheel import greedy_cover
  res = greedy_cover(list(range(1, 19)), max_notes=30, restarts=3, seed=42)
"""
from __future__ import annotations

import itertools
import math
import random
from dataclasses import dataclass
from typing import List, Optional, Sequence

# N>20 时 pass_rate 的抽样估计参数
SAMPLING_N = 20000
SAMPLING_ERROR = 0.0035  # ±0.35%: sqrt(0.25/20000) ≈ 0.00354
_MAX_NUM = 33            # 双色球红球域 1..33


@dataclass(frozen=True)
class CoverResult:
    """贪心覆盖的一次运行结果(不可变)。"""

    tickets: List[List[int]]          # 每注 6 个红球(升序, 来自 pool)
    n_notes: int
    covered_4subsets: int
    total_4subsets: int
    four_subset_coverage: float       # covered/total
    pass_rate: float                  # 6-子集通过率(精确计算, pool≤20)
    pass_rate_sampled: Optional[float]  # pool>20 时抽样估计, 否则 None
    converged: bool                   # 4-子集是否全覆盖(score=0 提前终止)
    max_notes: int


def _num_to_bit(n: int) -> int:
    """号码 n 编码为位掩码 bit = 1 << (n-1)。"""
    return 1 << (n - 1)


def _mask_of(nums: Sequence[int]) -> int:
    """一组号码 -> 位掩码。"""
    m = 0
    for n in nums:
        m |= 1 << (n - 1)
    return m


def greedy_cover(
    pool: Sequence[int],
    k: int = 6,
    t: int = 4,
    max_notes: int = 30,
    restarts: int = 3,
    seed: int = 0,
) -> CoverResult:
    """加权贪心集合覆盖: 生成至多 max_notes 注, 最大化 6-子集通过率。

    Args:
        pool: 红球池(1..33 内的号码, 自动去重排序)。
        k: 每注号码数(固定 6)。
        t: 覆盖子集大小(固定 4)。
        max_notes: 最大注数(≥ k)。
        restarts: 贪心重启次数(≥ 1), 每次不同 seed 打乱候选序, 按 pass_rate 取最优。
        seed: 随机种子, 全程固定保证可复现。

    Returns:
        CoverResult。

    Raises:
        ValueError: 参数非法(中文信息)。
    """
    # ---- 参数校验 (顺序执行, 全部 raise ValueError 带中文信息) ----
    pool = sorted(set(pool))
    if len(pool) < k:
        raise ValueError(f"红球池大小必须 ≥ k(={k}), 实际 {len(pool)}")
    for n in pool:
        if n < 1 or n > _MAX_NUM:
            raise ValueError(f"红球号码必须在 1-{_MAX_NUM} 之间, 实际 {n}")
    if not (1 <= t < k):
        raise ValueError(f"必须满足 1 ≤ t < k(={k}), 实际 t={t}")
    if max_notes < k:
        raise ValueError(f"max_notes 必须 ≥ k(={k}), 实际 {max_notes}")
    if restarts < 1:
        raise ValueError(f"restarts 必须 ≥ 1, 实际 {restarts}")
    if len(pool) > _MAX_NUM:
        raise ValueError(f"红球池大小不能超过 {_MAX_NUM}, 实际 {len(pool)}")

    N = len(pool)

    # ---- 1. 预计算: 6-子集 / 4-子集 位掩码与反向索引 ----
    k_choose_t = math.comb(k, t)  # 每 6-子集含 C(6,4)=15 个 4-子集
    # 枚举顺序固定(itertools.combinations 字典序), 保证确定性。
    six_subsets = list(itertools.combinations(pool, k))          # 号码元组(C(N,6) 个)
    q_masks = [_mask_of(c) for c in itertools.combinations(pool, t)]  # 全部 4-子集掩码
    q_index = {m: i for i, m in enumerate(q_masks)}
    num_q = len(q_masks)
    num_six = len(six_subsets)

    # 每个 6-子集 S 的 15 个 4-子集索引(与候选注同构: 候选 = 全部 6-子集)
    six_q_idx = []
    for s in six_subsets:
        six_q_idx.append([q_index[_mask_of(c)] for c in itertools.combinations(s, t)])

    # 反向索引: q -> [含 q 的 S 索引](每 q 有 C(N-4,2) 个 S)
    rev: List[List[int]] = [[] for _ in range(num_q)]
    for si, qs in enumerate(six_q_idx):
        for qi in qs:
            rev[qi].append(si)

    # 候选注 = 全部 6-子集, 候选序初始为字典序(后续按 seed 打乱)
    cand_order_base = list(range(num_six))

    def _run_once(restart_seed: int):
        """单次贪心运行: 返回 (chosen, covered, act, converged)。"""
        covered = [False] * num_q          # 4-子集是否已覆盖
        act = [0] * num_six                # 每 6-子集已覆盖的 4-子集数
        weight = [len(rev[qi]) for qi in range(num_q)]  # 含 q 的未激活 6-子集数
        order = cand_order_base[:]
        random.Random(restart_seed).shuffle(order)      # seed 打乱候选序, 确定性
        chosen: List[int] = []
        converged = False
        # 15 个 4-子集已全部覆盖的候选(score 恒 0, 扫描可跳过)
        full_cands = set()
        uncov_count = [k_choose_t] * num_six  # 每候选仍未覆盖的 4-子集数
        for _ in range(max_notes):
            best_score = 0
            best_ci = -1
            for ci in order:
                if ci in full_cands:
                    continue
                s = 0
                for qi in six_q_idx[ci]:
                    if not covered[qi]:
                        s += weight[qi]
                if s > best_score:      # 严格大于: 并列取候选序第一个
                    best_score = s
                    best_ci = ci
            if best_ci < 0 or best_score == 0:
                converged = True
                break
            chosen.append(best_ci)
            # 4. 更新: 该候选新覆盖的 4-子集
            new_qs = [qi for qi in six_q_idx[best_ci] if not covered[qi]]
            for qi in new_qs:
                covered[qi] = True
                for si in rev[qi]:
                    act[si] += 1
                    if act[si] == 1:    # 恰变 1: S 新激活
                        for q2 in six_q_idx[si]:
                            if q2 != qi and not covered[q2]:
                                weight[q2] -= 1
                    uncov_count[si] -= 1
                    if uncov_count[si] == 0:
                        full_cands.add(si)  # 该 S 全部 q 已覆盖, 后续跳过
        return chosen, covered, act, converged

    # ---- 5. restarts 次, 按 pass_rate 取最优 ----
    # N>20 时抽样 seed 固定(各 restart 同一样本, 对比公平)
    sample_rng = random.Random(seed + 99991) if N > 20 else None

    best: Optional[tuple] = None
    for r in range(restarts):
        chosen, covered, act, converged = _run_once(seed + r)
        passed = sum(1 for a in act if a > 0)
        if N <= 20:
            pr = passed / num_six
            pr_sampled: Optional[float] = None
        else:
            pr_sampled = _sample_pass_rate(pool, covered, q_masks, sample_rng)
            pr = pr_sampled
        if best is None or pr > best[4]:
            best = (chosen, covered, passed, act, pr, pr_sampled, converged)

    assert best is not None, "restarts ≥ 1 已校验, best 必有值"
    chosen, covered, passed, act, pr, pr_sampled, converged = best

    tickets = [list(six_subsets[ci]) for ci in chosen]
    total_4 = num_q
    covered_4 = sum(1 for c in covered if c)
    return CoverResult(
        tickets=tickets,
        n_notes=len(tickets),
        covered_4subsets=covered_4,
        total_4subsets=total_4,
        four_subset_coverage=covered_4 / total_4,
        pass_rate=pr,
        pass_rate_sampled=pr_sampled,
        converged=converged,
        max_notes=max_notes,
    )


def _sample_pass_rate(pool, covered: List[bool], q_masks: List[int], rng) -> float:
    """抽样估计 6-子集通过率: 随机 20000 个 6-子集, 任一 4-子集被覆盖即通过。

    抽样误差 ±0.35% (sqrt(p(1-p)/20000) ≤ 1/(2*sqrt(20000)) ≈ 0.00354)。
    """
    covered_set = {q_masks[i] for i, c in enumerate(covered) if c}
    hits = 0
    for _ in range(SAMPLING_N):
        nums = rng.sample(pool, 6)
        for comb in itertools.combinations(nums, 4):
            if _mask_of(comb) in covered_set:
                hits += 1
                break
    return hits / SAMPLING_N


def _self_check() -> None:
    """冒烟自检: 架构实测预期复现(池15/30注 ≥0.99, 池18 不崩且 ≥0.90)。"""
    r15 = greedy_cover(list(range(1, 16)), max_notes=30, restarts=3, seed=0)
    r18 = greedy_cover(list(range(1, 19)), max_notes=30, restarts=3, seed=0)
    print(f"pool=15: pass_rate={r15.pass_rate:.4f} (期望≥0.99) n_notes={r15.n_notes} "
          f"4子集覆盖={r15.four_subset_coverage:.4f} converged={r15.converged}")
    print(f"pool=18: pass_rate={r18.pass_rate:.4f} (期望≥0.90) n_notes={r18.n_notes} "
          f"4子集覆盖={r18.four_subset_coverage:.4f} converged={r18.converged}")


if __name__ == "__main__":
    _self_check()
