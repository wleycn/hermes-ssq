"""轻量 CRF 约束解码层（Dev-2）。

对概率池做贪心 beam 搜索，解码 k 个互异号码，且满足奇偶比/大小比约束
(odd_even ∈ {2:4, 3:3, 4:2}、size_ratio ∈ {2:4, 3:3, 4:2}, 1-16 小 / 17-33 大)。
全部候选不满足约束时退回概率 top-k；退化输入(空/NaN/全零/长度不足)不崩溃。

供 evaluate.py 的 crf 策略使用；生产 select_numbers 的 CRF 集成本次不做
(FEATURE_TOGGLES.crf 默认 false)。
"""
from typing import List, Optional, Sequence, Tuple

import numpy as np


def _odd_big(nums: Sequence[int]) -> Tuple[int, int]:
    """返回 (奇数个数, 大号个数(>16))。"""
    odds = sum(1 for x in nums if int(x) % 2 == 1)
    big = sum(1 for x in nums if int(x) > 16)
    return odds, big


def _feasible(odds: int, big: int, slots: int, pool: Sequence[int],
              odd_even: Tuple[int, ...], size_ratio: Tuple[int, ...]) -> bool:
    """部分注单(odds 个奇数, big 个大号, 还剩 slots 个号)能否补成满足约束的完整注。

    精确枚举剩余槽位的奇/偶与大/小可达范围, 判定是否存在合法完成方式。
    """
    if slots <= 0:
        return odds in odd_even and big in size_ratio
    # 剩余池中可用的奇/偶、大/小数量
    r_odd_big = sum(1 for x in pool if int(x) % 2 == 1 and int(x) > 16)
    r_odd_small = sum(1 for x in pool if int(x) % 2 == 1 and int(x) <= 16)
    r_even_big = sum(1 for x in pool if int(x) % 2 == 0 and int(x) > 16)
    r_even_small = sum(1 for x in pool if int(x) % 2 == 0 and int(x) <= 16)
    for odd_add in range(slots + 1):
        target_odd = odds + odd_add
        if target_odd not in odd_even:
            continue
        even_add = slots - odd_add
        if odd_add > r_odd_big + r_odd_small or even_add > r_even_big + r_even_small:
            continue
        # 可达大号数范围: 被迫选大的下限 vs 能选大的上限
        min_big = max(0, odd_add - r_odd_small, even_add - r_even_small)
        max_big = min(r_odd_big, odd_add) + min(r_even_big, even_add)
        if any(big + b in size_ratio for b in range(min_big, max_big + 1)):
            return True
    return False


def _top_k(pool: Sequence[int], k: int) -> List[int]:
    """退回路径：取概率池前 k 个(按传入顺序), 去重后升序返回, 不足 k 个则全返回。"""
    out: List[int] = []
    for x in pool:
        if int(x) not in out:
            out.append(int(x))
        if len(out) >= k:
            break
    return sorted(out)


def constrained_decode(probs: np.ndarray, k: int = 6, beam_width: int = 5,
                       odd_even: Tuple[int, ...] = (2, 3, 4),
                       size_ratio: Tuple[int, ...] = (2, 3, 4)) -> List[int]:
    """贪心 beam 约束解码。

    Args:
        probs: 号码概率数组, 长度即号码池大小(号码 = 下标+1, 如 33 -> 1..33)。
        k: 解码号码个数(默认 6)。
        beam_width: 每步保留的部分注单数。
        odd_even: 允许的奇数个数集合。
        size_ratio: 允许的大号(>16)个数集合。

    Returns:
        升序排列的 k 个互异号码; 退化输入返回 top-k(不崩溃)。
    """
    if k <= 0 or probs is None or len(probs) == 0:
        return []
    pool = [int(i) + 1 for i in range(len(probs))]
    if len(pool) < k:
        return sorted(pool)
    # 归一化/清洗: NaN/负值/全零均不崩溃
    p = np.nan_to_num(np.asarray(probs, dtype=np.float64), nan=0.0, posinf=0.0, neginf=0.0)
    p = np.maximum(p, 0.0)
    if p.sum() <= 0:
        return _top_k(pool, k)

    # 按概率降序的候选顺序
    order = [pool[i] for i in np.argsort(-p)]
    logp = {pool[i]: float(np.log(p[i] + 1e-300)) for i in range(len(pool))}

    # beam: 元素为 (累计对数概率, 部分注单list)
    beams: List[Tuple[float, List[int]]] = [(0.0, [])]
    for step in range(k):
        nxt: List[Tuple[float, List[int]]] = []
        for score, partial in beams:
            used = set(partial)
            odds, big = _odd_big(partial)
            slots_left = k - step - 1
            for cand in order:
                if cand in used:
                    continue
                n_odds, n_big = _odd_big([cand])
                new_partial = partial + [cand]
                if not _feasible(odds + n_odds, big + n_big, slots_left, pool, odd_even, size_ratio):
                    continue
                nxt.append((score + logp[cand], new_partial))
        if not nxt:
            break
        nxt.sort(key=lambda x: x[0], reverse=True)
        beams = nxt[:beam_width]

    # 完整且满足约束的候选中取概率最高者
    complete = [(s, t) for s, t in beams if len(t) == k]
    if complete:
        _, best = max(complete, key=lambda x: x[0])
        return sorted(best)
    # 无满足约束候选: 退回概率 top-k
    return _top_k(pool, k)
