#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""批量运行双色球预测模型，将各模型对红球(1-33)/蓝球(1-16)的预测概率
写入 PostgreSQL（schema: ssq），供 select_numbers 集成选号。

模型选择(均为 CPU 友好、秒级~分钟级, 规避慢 LSTM):
  - rf / lightgbm : 逐位置模型(Red1..Red6 + Blue1), 聚合为 33红/16蓝全量概率
  - cnn_reg  : 直接输出 all_red_probs(33) / all_blue_probs(16)

用法:
  .venv/bin/python batch_predict_pg.py                 # 跑全部模型 + 入库
  .venv/bin/python batch_predict_pg.py --model cnn_reg
"""
from __future__ import annotations
import argparse
import sys
import time
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

import numpy as np
import psycopg

# 性能铁律(2026-08-15 多agent独立验收确认):
#   torch.set_num_threads 是进程级全局设置, 必须在入口统一设 4。
#   默认 8 线程在 Ryzen7/8-vCPU 上跑 64 维小张量时调度开销爆炸:
#   LSTM +8818% / CNN +2860% / Transformer +591% (8线程 vs 4线程, 实测)。
#   放在这里而不是模型内部: MODELS 列表里 torch 模型顺序靠后, 若在
#   transformer train() 内才设置, 前面的 LSTM/CNN 已用 8 线程跑完。
try:
    import torch
    torch.set_num_threads(4)
except Exception:
    pass

import ml.main as M
from ml.config import RED_COLS, BLUE_COLS
from pg_schema import ensure_model_predictions_data_date

PG = dict(host="127.0.0.1", port=5432, user="hermes", password="hermes123", dbname="hermes")
SCHEMA = "ssq"
MODELS = ["rf", "lightgbm", "cnn_reg", "lstm", "transformer", "cdm"]


def ensure_schema(conn):
    with conn.cursor() as cur:
        cur.execute(f"CREATE SCHEMA IF NOT EXISTS {SCHEMA};")
        cur.execute(f"""
        CREATE TABLE IF NOT EXISTS {SCHEMA}.model_predictions (
            id          BIGSERIAL PRIMARY KEY,
            run_at      TIMESTAMP NOT NULL DEFAULT now(),
            model       TEXT NOT NULL,
            ball_type   TEXT NOT NULL,   -- 'red' | 'blue'
            num         INT  NOT NULL,   -- 1..33 (red) / 1..16 (blue)
            prob        DOUBLE PRECISION NOT NULL,
            UNIQUE (run_at, model, ball_type, num)
        );
        """)
        # 确保 data_date 列存在（ADR-3 / T3）：幂等迁移，存量行回填北京日期
        ensure_model_predictions_data_date(conn)
        conn.commit()


def aggregate_rf_results(res: dict):
    """rf/lgbm 的 batch_process 返回 {col: {all_numbers, all_probs}}。
    聚合成 33红/16蓝全量概率: 红球 n = 它在6个位置出现概率之和。
    res 键形如 'rf_Red1' / 'rf_Blue1' (带模型前缀)。"""
    reds = np.zeros(33)
    blues = np.zeros(16)
    for col, r in res.items():
        base = col.split("_", 1)[1] if "_" in col else col  # 去模型前缀
        nums = list(r.get("all_numbers", []))
        probs = list(r.get("all_probs", []))
        for n, p in zip(nums, probs):
            n = int(n)
            if base.startswith("Red") and 1 <= n <= 33:
                reds[n - 1] += float(p)
            elif base == "Blue1" and 1 <= n <= 16:
                blues[n - 1] += float(p)
    return reds, blues


def run_one(mt: str, df, retrain: bool = False):
    """训练+预测单个模型, 返回 (reds:33, blues:16) 或 None。
    支持部分输出模型:
      - cnn_reg / lstm / transformer : 红蓝全量 (all_red_probs + all_blue_probs)
      - rf / lightgbm               : 逐位置聚合 -> 红蓝全量
    缺失的那一侧返回 None, 由 load_latest_probs 在集成时跳过。

    retrain: True=重训+保存; False=load 已存模型(run_train 内部兜底:
    模型不存在时自动训练)。重训与否由调用方显式决定——月度重训由 cron
    (retrain_pipeline.py --no-email, 每月1号03:00) 负责, 本脚本不做
    mtime/日期自动判断(2026-08-16 Rocky 指示: 不需要在代码层面实现)。"""
    try:
        print(f"     模型状态: {'重训模式(--retrain)' if retrain else '复用已存模型(不存在则自动训练)'}")
        if mt in ("rf", "lightgbm"):
            res = M.batch_process(mt, df, RED_COLS + ["Blue1"], retrain)
            if not res:
                return None
            reds, blues = aggregate_rf_results(res)
            if reds.sum() == 0 or blues.sum() == 0:
                return None
            return reds, blues
        # 其余为 torch 模型
        M.run_train(mt, df, retrain)
        r = M.run_predict(mt, df)
        if not r:
            return None
        has_red = "all_red_probs" in r and len(r["all_red_probs"]) == 33
        has_blue = "all_blue_probs" in r and len(r["all_blue_probs"]) == 16
        if not has_red and not has_blue:
            print(f"[跳过] {mt} 无可用概率输出")
            return None
        reds = np.array(r["all_red_probs"], float)[:33] if has_red else None
        blues = np.array(r["all_blue_probs"], float)[:16] if has_blue else None
        return reds, blues
    except Exception as e:
        print(f"[跳过] {mt} 失败: {e!r}")
        return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=None)
    ap.add_argument("--retrain", action="store_true",
                    help="强制重训全部模型(月度 cron/手动重训用); 默认复用已存模型")
    args = ap.parse_args()

    print(f"[run] {datetime.now():%Y-%m-%d %H:%M:%S} 加载数据...")
    df = M.load_data()
    print(f"     数据行数: {len(df)}")

    targets = [args.model] if args.model else MODELS
    conn = psycopg.connect(**PG)
    ensure_schema(conn)
    run_at = datetime.now()

    for mt in targets:
        print(f"\n=== 模型 {mt} ===")
        t0 = time.time()
        out = run_one(mt, df, args.retrain)
        if out is None:
            continue
        reds, blues = out
        with conn.cursor() as cur:
            data_date = run_at.date()  # 北京日期（run_at 由 datetime.now() 生成），显式写入
            if reds is not None:
                for i, p in enumerate(reds, start=1):
                    cur.execute(
                        f"INSERT INTO {SCHEMA}.model_predictions(run_at,model,ball_type,num,prob,data_date) "
                        "VALUES (%s,%s,'red',%s,%s,%s) "
                        "ON CONFLICT (run_at,model,ball_type,num) DO UPDATE SET prob=EXCLUDED.prob;",
                        (run_at, mt, i, float(p), data_date))
            if blues is not None:
                for i, p in enumerate(blues, start=1):
                    cur.execute(
                        f"INSERT INTO {SCHEMA}.model_predictions(run_at,model,ball_type,num,prob,data_date) "
                        "VALUES (%s,%s,'blue',%s,%s,%s) "
                        "ON CONFLICT (run_at,model,ball_type,num) DO UPDATE SET prob=EXCLUDED.prob;",
                        (run_at, mt, i, float(p), data_date))
        conn.commit()
        sides = []
        if reds is not None: sides.append(f"红和={reds.sum():.2f}")
        if blues is not None: sides.append(f"蓝和={blues.sum():.2f}")
        print(f"     完成, 耗时 {time.time()-t0:.1f}s, {' '.join(sides)}")

    conn.close()
    print(f"\n[done] {datetime.now():%Y-%m-%d %H:%M:%S} 模型概率已写入 PG schema={SCHEMA}")


if __name__ == "__main__":
    main()
