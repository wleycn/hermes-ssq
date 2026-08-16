#!/usr/bin/env python3
"""红球大号区间弥散信号诊断 (2026-08-16)。

背景: ml/spectral_red.py 全量 3488 期探针 verdict=SCALAR_BIAS——
  路径2 子类 区间3[23..33] z=-3.006（近临界 3.254 未达）;
  路径3 和值均值 100.961 vs 精确 null 102.0 (z=-2.865, p=0.0042, 效应量 ~1%)。
结论模板标注"逐年代弥散、无实践意义"。本脚本做三件事验证该结论:

1. 时间切片: 全量分 4 段(2003-2008 / 2009-2014 / 2015-2020 / 2021-2026),
   每段重算 区间3 z + 和值均值 z —— 信号是稳定存在还是单一时期驱动?
2. 区间细分: 把 [23..33] 拆为 [23..28] 与 [29..33] 两段, 定位偏冷集中段。
3. 和值成因: 各段和值均值 vs 精确 null, 量化效应量。

复用 ml/spectral_red.py 纯函数(cooccurrence_matrix / subclass_stats /
exact_sum_null / moments_z_test), 不重复实现统计。输出:
  analysis/results/red_zone_drift.md

用法: .venv/bin/python analysis/red_zone_drift.py
"""
from __future__ import annotations

import itertools
import json
import math
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from ml.spectral_red import (  # noqa: E402
    cooccurrence_matrix,
    exact_sum_null,
    moments_z_test,
    subclass_stats,
)

RED_COLS = [f"Red{i}" for i in range(1, 7)]
OUT = ROOT / "analysis/results/red_zone_drift.md"

# 时间切片边界(含): 段1 2003-2008 / 段2 2009-2014 / 段3 2015-2020 / 段4 2021-2026
TIME_SLICES = [
    ("2003-2008", datetime(2003, 1, 1), datetime(2008, 12, 31)),
    ("2009-2014", datetime(2009, 1, 1), datetime(2014, 12, 31)),
    ("2015-2020", datetime(2015, 1, 1), datetime(2020, 12, 31)),
    ("2021-2026", datetime(2021, 1, 1), datetime(2026, 12, 31)),
]


def subset_pairs(subset: list[int]) -> tuple[int, int, int]:
    """子类统计参数: (n_pairs, n_shared, n_disjoint)。

    子集 S 大小 k: 对 P = C(k,2)。
      n_shared  = 每对共享 1 号的其他有向对-对计数 = C(k,2)*2*(k-2)
      n_disjoint = 每对不相交的其他有向对-对计数 = C(k,2)*C(k-2,2)
    验证: k=11 -> (55, 990, 1980) 与 SUBCLASS_TABLE 区间3 一致。
    """
    k = len(subset)
    n_pairs = math.comb(k, 2)
    n_shared = n_pairs * 2 * (k - 2)
    n_disjoint = n_pairs * math.comb(k - 2, 2)
    return n_pairs, n_shared, n_disjoint


def zone_stats(c: np.ndarray, subset: list[int], n_periods: int, name: str):
    """子集内全部号码对同现聚合 z 检验(复用 subclass_stats 精确矩闭式)。"""
    obs = 0
    for i, j in itertools.combinations(subset, 2):
        obs += int(c[i - 1, j - 1])
    npairs, nshared, ndisjoint = subset_pairs(subset)
    return subclass_stats(obs, npairs, nshared, ndisjoint, n_periods, name=name)


def sum_z(arr: np.ndarray):
    """和值均值 z 检验(精确卷积 null)。"""
    _, sum_mean, sum_var = exact_sum_null()
    return moments_z_test(float(arr.sum(axis=1).mean()), sum_mean, sum_var, len(arr))


def load_data() -> pd.DataFrame:
    df = pd.read_csv(ROOT / "ml/data/1.csv")
    df["dDate"] = pd.to_datetime(df["dDate"])
    for col in RED_COLS:
        df[col] = df[col].astype(int)
    return df


def analyze(df: pd.DataFrame, name: str, start: datetime, end: datetime) -> dict:
    sub = df[(df["dDate"] >= start) & (df["dDate"] <= end)]
    arr = sub[RED_COLS].to_numpy(dtype=int)
    n = len(arr)
    c = cooccurrence_matrix(arr)
    zone23 = zone_stats(c, list(range(23, 34)), n, "区间3[23..33]")
    zone_a = zone_stats(c, list(range(23, 29)), n, "[23..28]")
    zone_b = zone_stats(c, list(range(29, 34)), n, "[29..33]")
    s = sum_z(arr)
    return {
        "name": name, "n": n,
        "zone3": {"observed": zone23.observed, "expected": round(zone23.expected, 1),
                  "z": round(zone23.z, 3), "significant": zone23.significant},
        "zone_a": {"observed": zone_a.observed, "expected": round(zone_a.expected, 1),
                   "z": round(zone_a.z, 3), "significant": zone_a.significant},
        "zone_b": {"observed": zone_b.observed, "expected": round(zone_b.expected, 1),
                   "z": round(zone_b.z, 3), "significant": zone_b.significant},
        "sum": {"obs_mean": round(float(arr.sum(axis=1).mean()), 4),
                "z": round(s.z, 3), "p": round(s.p_value, 4),
                "significant": s.significant},
    }


def render(results: list[dict]) -> str:
    lines = [
        "# 红球大号区间弥散信号诊断 (2026-08-16)",
        "",
        "## 结论速览",
        "",
    ]
    full = results[0]
    lines += [
        f"- 全量 {full['n']} 期复现: 区间3[23..33] z={full['zone3']['z']} "
        f"(obs {full['zone3']['observed']} vs exp {full['zone3']['expected']}), "
        f"和值均值 z={full['sum']['z']} (p={full['sum']['p']})",
        "- 判据: 子类族 Bonferroni 临界 |z|=3.254(α_comp/5); 和值均值 z 用双侧 p<0.00568",
        "",
        "| 时段 | N | 区间3 z | [23..28] z | [29..33] z | 和值均值 | 和值 z | 和值 p |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for r in results:
        lines.append(
            f"| {r['name']} | {r['n']} | {r['zone3']['z']} | {r['zone_a']['z']} | "
            f"{r['zone_b']['z']} | {r['sum']['obs_mean']} | {r['sum']['z']} | {r['sum']['p']} |"
        )
    lines += ["", "## 判定", ""]
    # 精确口径: 区间3 子类用 Bonferroni 临界 |z|=3.254; 和值均值用双侧 p<0.00568
    zone3_sig = any(r["zone3"]["significant"] for r in results)
    sum_sig = any(r["sum"]["significant"] for r in results)
    if not zone3_sig and not sum_sig:
        lines.append("无任何时段/子段越过各自临界 -> 全量 SCALAR_BIAS 属弥散波动, "
                     "无稳定结构, 无选号含义。")
    else:
        if zone3_sig:
            for r in results:
                if r["zone3"]["significant"]:
                    lines.append(f"- {r['name']}: 区间3[23..33] z={r['zone3']['z']} 越 Bonferroni 临界 3.254")
        if sum_sig:
            for r in results:
                if r["sum"]["significant"]:
                    lines.append(f"- {r['name']}: 和值均值 z={r['sum']['z']} (p={r['sum']['p']} < 0.00568) 越界")
        lines.append("注意: 显著≠可预测, 需排除多重检验假阳性与数据源伪影。")
    lines.append("")
    return "\n".join(lines)


def main() -> None:
    df = load_data()
    results = []
    # 全量段(用全日期范围)
    results.append(analyze(df, "全量", datetime(2000, 1, 1), datetime(2100, 1, 1)))
    for name, start, end in TIME_SLICES:
        results.append(analyze(df, name, start, end))
    md = render(results)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(md, encoding="utf-8")
    print(md)
    print(f"\n[red_zone_drift] 报告 -> {OUT}")


if __name__ == "__main__":
    main()
