"""双色球预测系统 - 极简主入口 (ml/main.py)
    python -m ml.main train-predict-batch --m rf --cols all
    python -m ml.main train-predict-batch --m lgbm --cols all
    python -m ml.main train-predict --m lstm_all
    python -m ml.main train-predict --m lstm_blue
    python -m ml.main train-predict --m lstm_reds
    python -m ml.main train-predict --m cnn_math

"""
import argparse, json, logging, sys, time
from pathlib import Path
import numpy as np, pandas as pd
from sklearn.model_selection import train_test_split

# 假设这些从 ml.config 导入
from ml.config import ModelType, TARGET_COLS, RED_COLS, BLUE_COLS, DATA_FILE, MODELS_DIR, OUTPUT_DIR, MODEL_CONFIG, TRANSFORMER_CONFIG
from ml.data import load_data, extract_feature_columns
from ml.features import FeatureEngineer
from ml.utils.helpers import print_banner, build_prediction_dataframe

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s", handlers=[logging.StreamHandler(sys.stdout)])
logger = logging.getLogger("ssq.main")

_MODEL_MAP = {
    "rf": ("ml.models.rf_model", "RandomForestModel"),
    "lgbm": ("ml.models.lgb_model", "LightGBMModel"),
    "lstm_blue": ("ml.models.lstm_model", "LSTMBlueModel"),
    "lstm_reds": ("ml.models.lstm_model", "LSTMRedModel"),
    "lstm_all": ("ml.models.lstm_model", "LSTMAllModel"),
    "cnn_math": ("ml.models.cnn_model", "CNNMathModel"),
    "transformer_all": ("ml.models.transformer_model", "TransformerAllModel"),
    "cdm": ("ml.models.cdm_model", "CDMModel"),
}

def get_model_cls(mt):
    mod, cls = _MODEL_MAP[mt]
    return getattr(__import__(mod, fromlist=[cls]), cls)

def save_res(res, mt, col=None):
    d = Path(OUTPUT_DIR); d.mkdir(exist_ok=True)
    rows = []
    if "all_numbers" in res:
        for n, p in zip(res["all_numbers"], res["all_probs"]): rows.append({"Model": mt, "Ball": col or mt, "Num": n, "Prob": round(p, 6)})
    else:
        for k, prefix in [("all_red_probs", "Red"), ("all_blue_probs", "Blue")]:
            if k in res:
                for i, p in enumerate(res[k]): rows.append({"Model": mt, "Ball": prefix, "Num": i+1, "Prob": round(p, 6)})
    if rows: pd.DataFrame(rows).to_csv(d / f"pred_{mt}_{col or 'all'}_{time.strftime('%H%M%S')}.csv", index=False)

def run_train(mt, df, retrain=True, col=None):
    mc = get_model_cls(mt)
    mn = f"{mt}_{col}" if col else mt
    sd = MODELS_DIR / mn
    if not retrain and sd.exists():
        m = mc(mn); m.load(sd); return m
    
    print(f"🚀 开始训练: {mn}")
    # Transformer 使用 ml/config.py 的统一配置 (模块内默认 window=128 会被覆盖为优化值)
    if mt == "transformer_all":
        m = mc(mn, config=TRANSFORMER_CONFIG)
    else:
        m = mc(mn)
    
    if mt in ["rf", "lgbm"]:
        X, y, _, _ = m.prepare_data(df[col])
    elif mt == "cdm":
        # CDM 无监督频数模型: 直接喂原始开奖 DataFrame
        X, y = df, None
    elif "lstm" in mt or mt == "transformer_all":
        fe = FeatureEngineer(); dfe = fe.compute_all_features(df)
        fc = extract_feature_columns(dfe); ws = m.config.get("window_size", 128)
        args = {"df": dfe, "feature_cols": fc, "window_size": ws}
        if mt == "lstm_blue": args["target_col"] = "Blue1"
        elif mt == "lstm_reds": args["target_cols"] = RED_COLS
        else: args.update({"red_cols": RED_COLS, "blue_col": "Blue1"})
        X, y = m.prepare_data(**args)
        X, _, y, _ = train_test_split(X, y, test_size=0.5, random_state=42)
    elif mt == "cnn_math":
        fe = FeatureEngineer(); dfe = fe.compute_all_features(df)
        tc = RED_COLS + BLUE_COLS; rt = m.config.get("regression_targets", ["Next_Sum","Next_OddRatio","Next_BigRatio","Next_Hot_Count","Next_Cold_Count","Next_Max_Omission","Next_Avg_Omission"])
        if not all(c in dfe.columns for c in rt): dfe = fe.generate_regression_targets(dfe)
        yc = np.array([[r-(i+1) for i,r in enumerate(row[:6])] + [row[6]-1] for row in dfe[tc].values])
        yr = dfe[rt].values; fm = dfe[extract_feature_columns(dfe)].values.astype(np.float32)
        X, yc, yr = m.prepare_data(fm, yc, yr)
        X, _, yc, _, yr, _ = train_test_split(X, yc, yr, test_size=0.2, random_state=42)
    
    m.train(X, (yc, yr) if mt == "cnn_math" else y)
    m.save(sd)
    return m

def run_predict(mt, df, col=None):
    mc = get_model_cls(mt)
    mn = f"{mt}_{col}" if col else mt
    sd = MODELS_DIR / mn
    if not sd.exists(): raise FileNotFoundError(f"模型不存在: {sd}")
    
    print(f"🔮 开始预测: {mn}")
    m = mc(mn); m.load(sd)
    
    if mt in ["rf", "lgbm"]:
        ws = m.config.get("window_size", 165) - 1
        xl = pd.DataFrame(df[col].iloc[-(ws+1):-1].values.reshape(1, -1))
        if mt == "lgbm":
            from ml.models.lgb_model import calculate_all_features
            xl = calculate_all_features(xl)
            if hasattr(m, '_X_train_columns'): xl = xl.reindex(columns=m._X_train_columns, fill_value=0)
        else: xl.columns = m._X_train_columns if hasattr(m, '_X_train_columns') else xl.columns
        proba = m.predict_proba(xl)[0] if m.predict_proba(xl).ndim == 2 else m.predict_proba(xl)
        nums = m.label_encoder.inverse_transform(np.arange(len(proba)))
        return {"column": col, "top_numbers": nums[np.argsort(proba)[-6:][::-1]].tolist(), 
                "top_probs": np.sort(proba)[-6:][::-1].tolist(), "all_numbers": nums.tolist(), "all_probs": proba.tolist()}
    
    if mt == "cdm":
        # CDM 无监督频数模型: 无需特征, 直接后验均值
        p = m.predict_proba()[0]
        ri, bi = np.argsort(p[:33])[-6:][::-1], np.argsort(p[33:])[-6:][::-1]
        return {"top_red_numbers": (ri+1).tolist(), "top_red_probs": p[:33][ri].tolist(),
                "top_blue_numbers": (bi+1).tolist(), "top_blue_probs": p[33:][bi].tolist(),
                "all_red_probs": p[:33].tolist(), "all_blue_probs": p[33:].tolist()}

    fe = FeatureEngineer(); dfe = fe.compute_all_features(df)
    fc = getattr(m, 'feature_cols', None) or extract_feature_columns(dfe)
    ws = m.config.get("window_size", 128 if ("lstm" in mt or mt == "transformer_all") else 33)
    
    if "lstm" in mt or mt == "transformer_all":
        xd = dfe[fc].values.astype(np.float32)
        xl = np.zeros((1, ws, len(fc))); xl[0] = xd[-ws:]
        if m.scaler: xl = m.scaler.transform(xl.reshape(-1, xl.shape[-1])).reshape(xl.shape)
        
        if mt == "lstm_blue":
            p = m.predict_proba(xl)[0]; idx = np.argsort(p)[-6:][::-1]
            return {"top_blue_numbers": (idx+1).tolist(), "top_blue_probs": p[idx].tolist(), "all_blue_probs": p.tolist()}
        elif mt == "lstm_reds":
            p = m.predict_proba(xl)[0]; idx = np.argsort(p)[-6:][::-1]
            return {"top_red_numbers": (idx+1).tolist(), "top_red_probs": p[idx].tolist(), "all_red_probs": p.tolist()}
        else:
            rp, bp = m.predict_split(xl); rp, bp = rp[0], bp[0]
            ri, bi = np.argsort(rp)[-6:][::-1], np.argsort(bp)[-6:][::-1]
            return {"top_red_numbers": (ri+1).tolist(), "top_red_probs": rp[ri].tolist(), 
                    "top_blue_numbers": (bi+1).tolist(), "top_blue_probs": bp[bi].tolist(),
                    "all_red_probs": rp.tolist(), "all_blue_probs": bp.tolist()}
    
    elif mt == "cnn_math":
        tc = RED_COLS + BLUE_COLS; rt = m.config.get("regression_targets", ["Next_Sum","Next_OddRatio","Next_BigRatio","Next_Hot_Count","Next_Cold_Count","Next_Max_Omission","Next_Avg_Omission"])
        if not all(c in dfe.columns for c in rt): dfe = fe.generate_regression_targets(dfe)
        fm = dfe[fc].values.astype(np.float32)
        xl = fm[-(ws-1):].reshape(1, ws-1, -1)
        pred, reg = m.predict_with_post_processing(xl, dfe)
        rp_list, bp = m.predict_proba(xl)
        arp = np.zeros(33)
        for pos, probs in enumerate(rp_list):
            for idx, p in enumerate(probs): arp[idx + pos] += p
        ri, bi = np.argsort(arp)[-6:][::-1], np.argsort(bp)[-6:][::-1]
        return {"prediction": pred.tolist(), "reg_features": reg, 
                "top_red_numbers": (ri+1).tolist(), "top_red_probs": arp[ri].tolist(),
                "top_blue_numbers": (bi+1).tolist(), "top_blue_probs": bp[bi].tolist(),
                "all_red_probs": arp.tolist(), "all_blue_probs": bp.tolist()}

def batch_process(mt, df, cols, retrain):
    print(f"⚡ 批量处理: {mt}")
    res = {}
    need_col = mt in ["rf", "lgbm"]
    targets = cols if need_col else [None]
    
    for c in targets:
        try:
            m = run_train(mt, df, retrain, c)
            r = run_predict(mt, df, c)
            res[f"{mt}_{c}" if c else mt] = r
            save_res(r, mt, c)
        except Exception as e: logger.error(f"处理 {c} 失败: {e}")
    
    if need_col:
        print("\n📊 红球汇总:")
        for c in cols:
            k = f"{mt}_{c}"
            if k in res: print(f"  {c}: {res[k]['top_numbers']}")
    return res

def main():
    p = argparse.ArgumentParser(description="SSQ Predictor")
    sp = p.add_subparsers(dest="cmd")
    
    tp = sp.add_parser("train-predict"); tp.add_argument("-m", "--model", required=True, choices=list(_MODEL_MAP.keys()))
    tb = sp.add_parser("train-predict-batch"); tb.add_argument("-m", "--model", required=True, choices=["rf", "lgbm"])
    tb.add_argument("--cols", default="red", choices=["red", "blue", "all"])
    
    args = p.parse_args()
    if not args.cmd: p.print_help(); return
    
    df = load_data()
    if args.cmd == "train-predict":
        run_train(args.model, df, True)
        r = run_predict(args.model, df)
        save_res(r, args.model)
    else:
        cols = RED_COLS if args.cols == "red" else BLUE_COLS if args.cols == "blue" else TARGET_COLS
        batch_process(args.model, df, cols, True)

if __name__ == "__main__": main()

