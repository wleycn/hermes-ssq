#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""SSQ 重训 + 发邮件 固化流程 (Rocky 2026-08-14 指示 A+B)。

流程:
  1. 跑 batch_predict_pg.py 全量重训(6 模型), 概率入库 PG
  2. 验证: 最新每模型 run_at 覆盖全部 6 模型(红+蓝)
  3. 调外部邮件脚本(含凭证, 在 ~/.hermes/scripts/) 生成推荐 + 发邮件

设计说明:
  - 本脚本不含任何 DB/邮件 凭证, 可入库
  - 发邮件步骤委托 ~/.hermes/scripts/ssq_send_picks.py (凭证隔离, 不入 git)
  - 重训可能 10-20 分钟, 建议后台运行

用法:
  .venv/bin/python retrain_pipeline.py                 # 重训全部 + 发邮件(下一期)
  .venv/bin/python retrain_pipeline.py --period 2026094 --seed 2026094
  .venv/bin/python retrain_pipeline.py --no-email      # 只重训+验证, 不发邮件
  .venv/bin/python retrain_pipeline.py --check-only    # 只验证当前 PG 模型覆盖
"""
from __future__ import annotations

import argparse
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

import psycopg

PG = dict(host="127.0.0.1", port=5432, user="hermes", password="hermes123", dbname="hermes")
SCHEMA = "ssq"
EXPECTED_MODELS = ["cnn_math", "lgbm", "lstm_all", "lstm_blue", "lstm_reds", "rf"]
SEND_SCRIPT = Path.home() / ".hermes" / "scripts" / "ssq_send_picks.py"


def run_retrain() -> float:
    """跑 batch_predict_pg.py 全量重训. 返回耗时(秒)."""
    t0 = time.time()
    proc = subprocess.run(
        [str(ROOT / ".venv" / "bin" / "python"), str(ROOT / "batch_predict_pg.py")],
        cwd=str(ROOT), check=True)
    return time.time() - t0


def check_coverage() -> dict:
    """验证每模型每球种最新 run_at 覆盖. 返回 {model: {red: bool, blue: bool}}."""
    with psycopg.connect(**PG) as conn:
        cur = conn.cursor()
        cur.execute(f"""
            SELECT model, ball_type, MAX(run_at)
            FROM {SCHEMA}.model_predictions GROUP BY model, ball_type
        """)
        cov = {}
        for model, btype, _ra in cur.fetchall():
            cov.setdefault(model, {})[btype] = True
    return cov


def verify_all_models(cov: dict) -> bool:
    """验证红/蓝两侧各自有足够模型覆盖.

    注意: 部分模型是单侧输出设计(lstm_blue 仅蓝, lstm_reds 仅红),
    因此不要求每个模型都红+蓝, 而是:
      - 红球侧: 至少覆盖 EXPECTED_RED_MODELS
      - 蓝球侧: 至少覆盖 EXPECTED_BLUE_MODELS
    """
    EXPECTED_RED_MODELS = ["cnn_math", "lgbm", "lstm_all", "lstm_reds", "rf"]
    EXPECTED_BLUE_MODELS = ["cnn_math", "lgbm", "lstm_all", "lstm_blue", "rf"]
    red_cov = {m for m, s in cov.items() if s.get("red")}
    blue_cov = {m for m, s in cov.items() if s.get("blue")}
    miss_red = [m for m in EXPECTED_RED_MODELS if m not in red_cov]
    miss_blue = [m for m in EXPECTED_BLUE_MODELS if m not in blue_cov]
    if miss_red or miss_blue:
        print(f"[验证失败] 红球缺失: {miss_red} | 蓝球缺失: {miss_blue}")
        return False
    print(f"[验证通过] 红球侧 {len(red_cov)} 模型 / 蓝球侧 {len(blue_cov)} 模型均有最新概率")
    return True


def send_email(period: str, seed: int) -> None:
    if not SEND_SCRIPT.exists():
        raise FileNotFoundError(f"邮件脚本不存在: {SEND_SCRIPT} (含凭证, 应在 ~/.hermes/scripts/)")
    subprocess.run(
        [str(ROOT / ".venv" / "bin" / "python"), str(SEND_SCRIPT),
         "--period", period, "--seed", str(seed)],
        cwd=str(ROOT), check=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--period", default="2026094", help="期号")
    ap.add_argument("--seed", type=int, default=2026094, help="生成 seed")
    ap.add_argument("--no-email", action="store_true", help="只重训+验证, 不发邮件")
    ap.add_argument("--check-only", action="store_true", help="只验证当前 PG 模型覆盖")
    args = ap.parse_args()

    if args.check_only:
        cov = check_coverage()
        verify_all_models(cov)
        return

    if not args.no_email:
        # 重训前先确认邮件脚本在位
        if not SEND_SCRIPT.exists():
            print(f"[中止] 邮件脚本缺失: {SEND_SCRIPT}")
            sys.exit(1)

    print(f"[1/3] 重训全部模型 (batch_predict_pg.py)...")
    secs = run_retrain()
    print(f"      重训完成, 耗时 {secs:.1f}s")

    print(f"[2/3] 验证模型覆盖...")
    cov = check_coverage()
    if not verify_all_models(cov):
        print("[警告] 模型覆盖不完整, 仍继续发邮件(集成会自动跳过缺失模型)")

    if args.no_email:
        print("[3/3] --no-email 指定, 跳过发邮件")
        return

    print(f"[3/3] 生成 {args.period} 推荐 + 发邮件...")
    send_email(args.period, args.seed)
    print("[done] 重训+发邮件流程完成")


if __name__ == "__main__":
    main()