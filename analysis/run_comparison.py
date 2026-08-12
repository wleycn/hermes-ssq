"""SSQ 对比测试驱动：旧分位置建模 vs 新集合建模，walk-forward 滚动外推。

用法:
  python analysis/run_comparison.py            # 全量(horizon=120)
  python analysis/run_comparison.py --smoke    # 冒烟(horizon=5, 快速验证链路)
结果:
  analysis/results/comparison_table.csv
  analysis/results/wf_<model>.json  (每期明细)
"""
import argparse
import json
import sys
import time
from pathlib import Path

# 确保项目根(含 ml 包)在 sys.path, 使直接 `python analysis/run_comparison.py` 也能 import ml
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

import numpy as np
import pandas as pd

from ml.data import load_data
from ml.config import RED_COLS, BLUE_COLS
from ml.features.feature_engineer import FeatureEngineer
from ml.models import (
    RandomForestModel, LightGBMModel,
    LSTMRedModel, CNNMathModel, SetRedModel,
)
from ml.eval.walk_forward import (
    run_walk_forward, freq_red_baseline_overlap,
    random_red_overlap_actual,
)

WS = 100  # 序列窗口


# ---------------- 旧模型适配 (分位置建模) ----------------
def legacy_red_predict(model, df_trunc, t, model_type):
    """旧模型: 用各自 prepare_data, 预测6个位置 top1 拼成集合。"""
    if model_type == "rf":
        # RF 在 main.py 里是分列训练; 这里简化为取每列最近窗口
        # 为公平且简单, 用最近一期各位置频率近似(见说明)
        return _rf_legacy_predict(df_trunc), 0
    # lstm_reds / cnn_math 走原 prepare_data
    return _lstm_cnn_legacy_predict(model, df_trunc, t, model_type)


def _rf_legacy_predict(df_trunc):
    # RF 分位置模型需要逐列训练, 这里仅做轻量近似: 每位置最近50期众数
    out = []
    for col in RED_COLS:
        vc = df_trunc[col].value_counts()
        out.append(int(vc.index[0]) if len(vc) else 1)
    return sorted(set(out))[:6] if len(set(out)) >= 6 else sorted(range(1, 7))


def _lstm_cnn_legacy_predict(model, df_trunc, t, model_type):
    from ml.features.feature_engineer import FeatureEngineer
    fe = FeatureEngineer()
    dfe = fe.compute_all_features(df_trunc).dropna()
    # 优先用模型训练时记忆的 feature_cols, 保证列数/顺序一致; 否则 fallback 数值列
    fc = getattr(model, "feature_cols", None)
    if not fc:
        fc = [c for c in dfe.select_dtypes(include=[np.number]).columns
              if c not in ["Red1","Red2","Red3","Red4","Red5","Red6","Blue1","Sum","Odd_Count"]]
    ws = model.config.get("window_size", 128)
    X = dfe[fc].values.astype(np.float32)
    xl = np.zeros((1, ws, len(fc)))
    xl[0] = X[-ws:]
    if model.scaler:
        xl = model.scaler.transform(xl.reshape(-1, xl.shape[-1])).reshape(xl.shape)
    if model_type == "lstm_reds":
        p = model.predict_proba(xl)[0]
        idx = np.argsort(p)[-6:][::-1]
        return (idx + 1).tolist(), 0
    else:  # cnn_math: 用后处理(去泄漏版, 传训练期统计)
        # 返回约定: (np.array(reds+[blue]), reg_dict) — 蓝球在 rp[6]
        rp, _ = model.predict_with_post_processing(xl, df_trunc, train_stats=getattr(model, "_train_stats", None))
        return list(rp[:6].astype(int)), int(rp[6])


# ---------------- 新模型: SetRed (统一特征集合预测) ----------------
def setred_factory(df_trunc):
    model = SetRedModel("set_red")
    fe = FeatureEngineer()
    dfe = fe.build_unified_features(df_trunc)
    fc = list(dfe.columns)
    if len(dfe) <= WS + 2:
        # 数据不足窗口, 退化为频率预测
        return _freq_model(dfe)
    X, y = model.prepare_data(dfe, df_trunc, feature_cols=fc, window_size=WS, target_cols=RED_COLS)
    # 切分 val
    n = len(X)
    va = max(1, n // 5)
    model.train(X[:-va], y[:-va], X[-va:], y[-va:], epochs=30)
    return model


def _freq_model(dfe):
    """退化模型: 直接存频率, predict 返回最频繁6号。"""
    class _Freq:
        def __init__(self, dfe):
            self.dfe = dfe
        def predict_proba(self, X):
            cnt = dfe[RED_COLS].stack().value_counts()
            top = [int(x) - 1 for x, _ in cnt.head(6).items()]
            p = np.zeros(33)
            for i in top:
                p[i] = 1.0
            return p[None, :]
    return _Freq(dfe)


def setred_predict(model, df_trunc, t):
    fe = FeatureEngineer()
    dfe = fe.build_unified_features(df_trunc)
    fc = list(dfe.columns)
    if len(dfe) <= WS + 2:
        p = model.predict_proba(np.zeros((1, 1, len(fc))))
        idx = np.argsort(p[0])[-6:][::-1]
        return (idx + 1).tolist(), 0
    X = dfe[fc].values.astype(np.float32)
    if hasattr(model, "scaler") and model.scaler:
        Xs = model.scaler.transform(X.reshape(-1, X.shape[-1])).reshape(X.shape)
    else:
        Xs = X
    xl = Xs[-WS:][None, :, :]
    p = model.predict_proba(xl)[0]
    idx = np.argsort(p)[-6:][::-1]
    return (idx + 1).tolist(), 0


# ---------------- 主流程 ----------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true", help="小样本冒烟测试(horizon=3)")
    ap.add_argument("--horizon", type=int, default=None, help="外推期数(默认120; smoke时3)")
    args = ap.parse_args()

    horizon = args.horizon if args.horizon is not None else (3 if args.smoke else 120)
    train_min = 800
    out_dir = Path("analysis/results")
    out_dir.mkdir(parents=True, exist_ok=True)

    df = load_data()
    print(f"数据期数: {len(df)}")

    results = {}

    # --- 新模型: SetRed (统一特征集合预测) ---
    t0 = time.time()
    print("\n[1/1] 训练+滚动预测 SetRed (新集合建模) ...")
    res_new = run_walk_forward(
        setred_factory, df, train_min=train_min, horizon=horizon,
        predict_fn=setred_predict,
    )
    results["set_red(unified)"] = res_new
    print(f"    红球平均集合命中={res_new['red_mean_overlap']:.3f}  蓝球top1={res_new['blue_top1_acc']:.3f}  ({time.time()-t0:.0f}s)")
    json.dump(res_new["details"], open(out_dir / "wf_set_red.json", "w"), default=str)

    # --- 旧模型 (分位置) 简化接入 ---
    # 为控制时长, 旧模型用轻量近似(位置众数/原prepare_data轻量), 重点对比"方法"
    # RF 用位置众数近似
    t0 = time.time()
    print("\n[2/4] RF 旧分位置建模(位置众数近似) ...")
    res_rf = run_walk_forward(
        lambda d: _rf_legacy_predict(d), df, train_min=train_min, horizon=horizon,
        predict_fn=lambda m, d, t: (m, 0),
    )
    results["rf(legacy-approx)"] = res_rf
    print(f"    红球平均集合命中={res_rf['red_mean_overlap']:.3f}  ({time.time()-t0:.0f}s)")

    # LSTM_REDS
    t0 = time.time()
    print("\n[3/4] LSTM_REDS 旧分位置建模 ...")
    def lstm_factory(d):
        m = LSTMRedModel("lstm_reds")
        fe = FeatureEngineer()
        dfe = fe.compute_all_features(d).dropna()
        fc = [c for c in dfe.select_dtypes(include=[np.number]).columns
              if c not in ["Red1","Red2","Red3","Red4","Red5","Red6","Blue1","Sum","Odd_Count"]]
        X, y = m.prepare_data(dfe, fc, target_cols=RED_COLS, window_size=WS)
        Xtr, Xva, ytr, yva = X[:-50], X[-50:], y[:-50], y[-50:]
        m.train(Xtr, ytr, Xva, yva)
        return m
    res_lstm = run_walk_forward(
        lstm_factory, df, train_min=train_min, horizon=horizon,
        predict_fn=lambda m, d, t: _lstm_cnn_legacy_predict(m, d, t, "lstm_reds"),
    )
    results["lstm_reds(legacy)"] = res_lstm
    print(f"    红球平均集合命中={res_lstm['red_mean_overlap']:.3f}  ({time.time()-t0:.0f}s)")
    json.dump(res_lstm["details"], open(out_dir / "wf_lstm_reds.json", "w"), default=str)

    # CNN_MATH
    t0 = time.time()
    print("\n[4/4] CNN_MATH 旧建模 ...")
    def cnn_factory(d):
        m = CNNMathModel("cnn_math")
        fe = FeatureEngineer()
        dfe = fe.compute_all_features(d).dropna()
        tc = RED_COLS + BLUE_COLS
        # 诚实回归目标: 当前期可观测统计量(非 Next_* shift 泄漏)
        rc = ["Sum","OddRatio","BigRatio","Hot_Count","Cold_Count","Max_Omission","Avg_Omission"]
        # 无条件生成回归目标列(含 Norm_Mean/Std/Poisson_* 等当前期统计, 非泄漏), 供训练期统计后处理用
        dfe = fe.generate_regression_targets(dfe)
        yc = np.array([[int(r)-(i+1) for i,r in enumerate(row[:6])] + [int(row[6])-1] for row in dfe[tc].values]).astype(np.int64)
        yr = dfe[rc].values.astype(np.float32)
        # 仅取数值列作特征矩阵(排除 dDate/dNum 等字符串与标签/回归目标)
        fm_cols = [c for c in dfe.select_dtypes(include=[np.number]).columns
                   if c not in ["Red1","Red2","Red3","Red4","Red5","Red6","Blue1",
                                "Sum","OddRatio","BigRatio","Hot_Count","Cold_Count","Max_Omission","Avg_Omission"]]
        fm = dfe[fm_cols].values.astype(np.float32)
        # 备份训练期统计, 供 predict_with_post_processing(train_stats=...) 去泄漏
        m._train_stats = {
            "norm_mean": dfe["Norm_Mean"].iloc[-1] if "Norm_Mean" in dfe.columns else np.nan,
            "norm_std": dfe["Norm_Std"].iloc[-1] if "Norm_Std" in dfe.columns else np.nan,
            "poisson_r": dfe[[f"Poisson_R{i}" for i in range(1,34)]].iloc[-1].values if all(f"Poisson_R{i}" in dfe.columns for i in range(1,34)) else [np.nan]*33,
            "poisson_b": dfe[[f"Poisson_B{i}" for i in range(1,17)]].iloc[-1].values if all(f"Poisson_B{i}" in dfe.columns for i in range(1,17)) else [np.nan]*16,
        }
        m.feature_cols = fm_cols  # 记住训练用特征列, 预测时保持一致
        X, yc, yr = m.prepare_data(fm, yc, yr)
        n = len(X)
        va = max(1, n // 5)
        m.train(X[:-va], (yc[:-va], yr[:-va]))
        return m
    res_cnn = run_walk_forward(
        cnn_factory, df, train_min=train_min, horizon=horizon,
        predict_fn=lambda m, d, t: _lstm_cnn_legacy_predict(m, d, t, "cnn_math"),
    )
    results["cnn_math(legacy)"] = res_cnn
    print(f"    红球平均集合命中={res_cnn['red_mean_overlap']:.3f} 蓝球top1={res_cnn['blue_top1_acc']:.3f}  ({time.time()-t0:.0f}s)")
    json.dump(res_cnn["details"], open(out_dir / "wf_cnn_math.json", "w"), default=str)

    # --- 基线 ---
    start, end = train_min, min(len(df) - 1, train_min + horizon)
    fr = freq_red_baseline_overlap(df, start, end, train_min)
    rr = random_red_overlap_actual(df, start, end)
    print(f"\n[基线] 频率基线集合命中={fr:.3f}  随机基线集合命中={rr:.3f}  (理论期望=1.09)")

    # --- 对比表 ---
    rows = []
    for name, r in results.items():
        rows.append({
            "model": name, "mode": "new" if "set_red" in name else "legacy",
            "red_mean_overlap": round(r["red_mean_overlap"], 4),
            "red_hit_ge3": round(r["red_hit_ge3"], 4),
            "blue_top1_acc": round(r["blue_top1_acc"], 4),
            "n_periods": r["n_periods"],
        })
    rows.append({"model": "freq_baseline", "mode": "baseline",
                 "red_mean_overlap": round(fr, 4), "red_hit_ge3": "", "blue_top1_acc": "", "n_periods": end-start+1})
    rows.append({"model": "random_baseline", "mode": "baseline",
                 "red_mean_overlap": round(rr, 4), "red_hit_ge3": "", "blue_top1_acc": "", "n_periods": end-start+1})
    pd.DataFrame(rows).to_csv(out_dir / "comparison_table.csv", index=False)
    print(f"\n✅ 对比表已存: {out_dir / 'comparison_table.csv'}")
    print(pd.DataFrame(rows).to_string(index=False))


if __name__ == "__main__":
    main()
