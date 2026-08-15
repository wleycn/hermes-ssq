#!/usr/bin/env python3
"""混沌/相空间重构检验 CLI — 双色球开奖序列随机性验证

用法:
    .venv/bin/python ml/spectral_chaos_cli.py            # 默认检验红球序列
    .venv/bin/python ml/spectral_chaos_cli.py --ball blue # 检验蓝球序列
    .venv/bin/python ml/spectral_chaos_cli.py --shuffle   # 先打乱每期排序再检验(打破排序伪影)

输出: 判定 (RANDOM / CHAOTIC/STRUCTURED) + Lyapunov z/p + 样本熵 z/p
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from ml.spectral_chaos import run_chaos_test

RED_COLS = ["Red1", "Red2", "Red3", "Red4", "Red5", "Red6"]


def main() -> None:
    ap = argparse.ArgumentParser(description="混沌/相空间重构检验器")
    ap.add_argument("--data", default="ml/data/1.csv", help="开奖数据 CSV 路径")
    ap.add_argument("--ball", choices=["red", "blue"], default="red", help="检验红球还是蓝球序列")
    ap.add_argument("--shuffle", action="store_true", help="每期内打乱排序后再检验(打破排序伪影)")
    ap.add_argument("--surrogates", type=int, default=100, help="替代序列数量")
    ap.add_argument("--seed", type=int, default=99, help="打乱用随机种子")
    args = ap.parse_args()

    df = pd.read_csv(args.data)
    cols = RED_COLS if args.ball == "red" else ["Blue1"]
    data = df[cols].values.astype(int)

    if args.shuffle:
        rng = np.random.default_rng(args.seed)
        data = np.array([rng.permutation(row) for row in data])

    series = data.flatten().astype(float)
    print(f"序列: {'红球(每期6个)' if args.ball == 'red' else '蓝球'} | "
          f"{len(df)} 期 | 长度 {len(series)} | {'打乱排序' if args.shuffle else '原始顺序'}")

    res = run_chaos_test(series, n_surrogates=args.surrogates)
    lyap = res.get("lyap", {})
    se = res.get("sampen", {})
    print(f"\nLyapunov (Rosenstein): 原始={lyap.get('value', float('nan')):.4f} "
          f"替代={lyap.get('surr_mean', float('nan')):.4f}±{lyap.get('surr_std', float('nan')):.4f} "
          f"z={lyap.get('z_score', 0):.2f} p={lyap.get('p_value', 1):.4f} -> {lyap.get('verdict', '?')}")
    print(f"样本熵: 原始={se.get('value', float('nan')):.4f} "
          f"替代={se.get('surr_mean', float('nan')):.4f}±{se.get('surr_std', float('nan')):.4f} "
          f"z={se.get('z_score', 0):.2f} p={se.get('p_value', 1):.4f} -> {se.get('verdict', '?')}")
    print(f"\n综合判定: {res.get('verdict', '?')}")
    print(f"说明: {res.get('note', '')}")
    print("\n提示: 若原始序列判 CHAOTIC 但 --shuffle 后变 RANDOM, 则结论是排序伪影, 实际无混沌结构")


if __name__ == "__main__":
    main()
