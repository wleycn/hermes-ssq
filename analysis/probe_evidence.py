#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""真实数据证据脚本: MF-DFA / RMT / 校准三探针跑 1.csv 全量数据, 结果落盘。

背景 (2026-08-22 tech-writer 审计 R3): ARCHITECTURE 决策记录中的
"红 H(2)=0.473/蓝 0.493、max_ratio z=-0.81、ECE≈0.007" 等真实数据数值
此前只在一次未落盘的交互运行中出现, 仓库内无可复现来源。本脚本补上:
每次运行把结果写入 analysis/results/probe_evidence_YYYY-MM-DD.txt,
使决策记录中的数值可被任何后续会话复现/复核。

用法:
  .venv/bin/python analysis/probe_evidence.py
  .venv/bin/python analysis/probe_evidence.py --n-surrogates 30
"""
from __future__ import annotations

import argparse
import csv
import sys
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import numpy as np

from ml.probes.calibration_probe import evaluate_calibration
from ml.probes.mfdfa_probe import run_mfdfa_probe
from ml.probes.rmt_probe import run_rmt_probe


def load_draws() -> tuple[list[list[int]], list[int]]:
    """读 1.csv, 返回 (红球每期6个列表, 蓝球列表)。"""
    reds: list[list[int]] = []
    blues: list[int] = []
    csv_path = ROOT / "ml/data/1.csv"
    with csv_path.open("r", encoding="utf-8-sig", newline="") as f:
        for row in csv.DictReader(f):
            reds.append([int(row[f"Red{i}"]) for i in range(1, 7)])
            blues.append(int(row["Blue1"]))
    return reds, blues


def run_evidence(n_surrogates: int = 30) -> str:
    reds, blues = load_draws()
    n = len(reds)
    red_flat = np.array([v for r in reds for v in r], dtype=float)
    blue_arr = np.array(blues, dtype=float)
    lines: list[str] = []
    lines.append(f"=== SSQ 真实数据探针证据 {datetime.now():%Y-%m-%d %H:%M} ===")
    lines.append(f"数据源: ml/data/1.csv, 共 {n} 期, 红球展平 {red_flat.size} 点, 蓝球 {blue_arr.size} 点")
    lines.append("")

    # 1) MF-DFA
    lines.append("[1] MF-DFA 多重分形 (红球展平序列 / 蓝球序列):")
    for label, seq in [("红球", red_flat), ("蓝球", blue_arr)]:
        for r in run_mfdfa_probe(seq, n_surrogates=n_surrogates):
            lines.append(f"  {label} {r.name}: value={r.value:.4f} {r.verdict} {r.detail}")
    lines.append("")

    # 2) RMT (33 号码 × 200 期窗口)
    lines.append(f"[2] RMT 随机矩阵谱 (33 号码 × 200 期窗口, surrogate n={n_surrogates}):")
    for r in run_rmt_probe(reds, n_surrogates=n_surrogates):
        lines.append(f"  {r.name}: value={r.value:.4f} {r.verdict} {r.detail}")
    lines.append("")

    # 2b) 08-22 补落地探针 (可见图/LZ/RQA/MSE/Rényi/DCCA)
    #     适用性: 可见图/RQA 用蓝球与和值(i.i.d. 序列); 红全量有组合约束伪影
    from ml.probes.visibility_probe import run_visibility_probe
    from ml.probes.lz_probe import run_lz_probe
    from ml.probes.rqa_probe import run_rqa_probe
    from ml.probes.mse_probe import run_mse_probe
    from ml.probes.renyi_probe import run_renyi_probe
    from ml.probes.dcca_probe import run_dcca_probe
    sums = np.array([sum(r) for r in reds], dtype=float)
    lines.append("[2b] 08-22 补落地探针 (可见图/LZ/RQA/MSE/Rényi/DCCA):")
    for label, seq in [("蓝球", blue_arr), ("和值", sums)]:
        lines.append(f"  --- {label} ---")
        for r in run_visibility_probe(seq, n_surrogates=15):
            lines.append(f"  可见图 {r.name}: value={r.value:.4f} {r.verdict} {r.detail}")
        for r in run_lz_probe(seq, n_surrogates=15):
            lines.append(f"  LZ {r.name}: value={r.value:.4f} {r.verdict} {r.detail}")
        for r in run_rqa_probe(seq, n_surrogates=15):
            lines.append(f"  RQA {r.name}: value={r.value:.4f} {r.verdict} {r.detail}")
        for r in run_mse_probe(seq):
            lines.append(f"  MSE {r.name}: value={r.value:.4f} {r.verdict} {r.detail}")
    # Rényi 只对类别序列(蓝球)跑; 和值是连续量非类别
    for r in run_renyi_probe(blue_arr, n_classes=16):
        lines.append(f"  Rényi {r.name}: value={r.value:.4f} {r.verdict} {r.detail}")
    lines.append("  DCCA (红球位1 vs 位2):")
    p0 = np.array([r[0] for r in reds], dtype=float)
    p1 = np.array([r[1] for r in reds], dtype=float)
    for r in run_dcca_probe(p0, p1, n_surrogates=15):
        lines.append(f"    {r.name}: value={r.value:.4f} {r.verdict} {r.detail}")
    lines.append("")

    # 3) 校准诊断 (用历史频率+噪声模拟模型概率, 对真实开奖评估 Brier/ECE)
    #    口径同 docs/steiner_walkforward_2026-08-22.md (W=500 前窗 + 高斯噪声)
    rng = np.random.default_rng(42)
    W = 500
    ph_red: list[np.ndarray] = []
    ph_blue: list[np.ndarray] = []
    outs_red: list[int] = []
    outs_blue: list[int] = []
    for t in range(W, n - 1):
        hist = reds[t - W:t]
        f_r = np.zeros(33)
        for row in hist:
            for v in row:
                f_r[v - 1] += 1
        p_r = f_r / f_r.sum() + rng.normal(0, 0.01, 33)
        p_r = np.clip(p_r, 1e-6, None)
        ph_red.append(p_r / p_r.sum())
        # 蓝球: 近似独立同分布, 用频率
        f_b = np.zeros(16)
        for v in blues[t - W:t]:
            f_b[v - 1] += 1
        p_b = f_b / f_b.sum() + rng.normal(0, 0.02, 16)
        p_b = np.clip(p_b, 1e-6, None)
        ph_blue.append(p_b / p_b.sum())
        outs_red.append(reds[t][0])          # 仅取首号做单点对照(简化)
        outs_blue.append(blues[t])
    lines.append(f"[3] 校准诊断 (模拟模型输出 vs 真实开奖, n={len(ph_red)}):")
    for label, ph, outs in [("红球", ph_red, outs_red), ("蓝球", ph_blue, outs_blue)]:
        for r in evaluate_calibration(ph, outs):
            lines.append(f"  {label} {r.metric}: {r.value} {r.detail}")
    lines.append("")
    lines.append("判据: MF-DFA H(2)≈0.5 且 Δh≈0 → RANDOM; RMT |z|<2 → RANDOM;")
    lines.append("      校准 ECE≈0.007 且 isotonic≈0 → i.i.d. 下校准无操作空间。")
    lines.append("结论预期: 全部 FLAT/RANDOM (与决策记录 2026-08-22 条目一致)。")
    return "\n".join(lines)


def main() -> None:
    ap = argparse.ArgumentParser(description="SSQ 真实数据探针证据落盘")
    ap.add_argument("--n-surrogates", type=int, default=30, help="surrogate 数量(默认 30)")
    args = ap.parse_args()
    text = run_evidence(args.n_surrogates)
    print(text)
    out_dir = ROOT / "analysis/results"
    out_dir.mkdir(parents=True, exist_ok=True)
    out = out_dir / f"probe_evidence_{datetime.now():%Y-%m-%d}.txt"
    out.write_text(text, encoding="utf-8")
    print(f"\n已落盘: {out}")


if __name__ == "__main__":
    main()
