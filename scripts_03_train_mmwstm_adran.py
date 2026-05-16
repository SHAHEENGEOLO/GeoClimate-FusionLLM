#!/usr/bin/env python3
"""Step 3: Train MMWSTM-ADRAN+ multi-stream ensemble."""
import numpy as np, pandas as pd, os, sys, json, warnings
warnings.filterwarnings('ignore')
sys.path.insert(0, os.path.dirname(__file__))
from config import *
from sklearn.preprocessing import RobustScaler
from sklearn.ensemble import (HistGradientBoostingRegressor, ExtraTreesRegressor,
                               RandomForestRegressor, GradientBoostingRegressor)
from sklearn.neural_network import MLPRegressor
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error
from scipy import optimize

def main():
    print("Loading data and previous results...")
    df = pd.read_csv(FEATURES_FILE, parse_dates=[DATE_COL])
    with open(RESULTS_FILE) as f: prev = json.load(f)
    feat_cols = prev['feature_cols']
    
    si = int(len(df) * TRAIN_SPLIT_RATIO)
    TR, TE = df.iloc[:si], df.iloc[si:]
    Xtr = TR[feat_cols].values.astype(np.float32)
    ytr = TR['target_next'].values.astype(np.float32)
    Xte = TE[feat_cols].values.astype(np.float32)
    yte = TE['target_next'].values.astype(np.float32)
    sc = RobustScaler(); Xtr_s = sc.fit_transform(Xtr); Xte_s = sc.transform(Xte)
    
    # --- 5 STREAMS ---
    models = {}
    print("  [1/5] MMWSTM-Deep (HistGBRT depth=6)...")
    models['S1'] = HistGradientBoostingRegressor(**STREAM1_MMWSTM_DEEP, random_state=42)
    models['S1'].fit(Xtr, ytr)
    
    print("  [2/5] MMWSTM-Wide (HistGBRT depth=4)...")
    models['S2'] = HistGradientBoostingRegressor(**STREAM2_MMWSTM_WIDE, random_state=123)
    models['S2'].fit(Xtr, ytr)
    
    print("  [3/5] ADRAN (DNN 512→256→128→64)...")
    models['S3'] = MLPRegressor(**STREAM3_ADRAN, random_state=42)
    models['S3'].fit(Xtr_s, ytr)
    
    print("  [4/5] ExtraTrees (400 trees)...")
    models['S4'] = ExtraTreesRegressor(**STREAM4_EXTRATREES, random_state=42)
    models['S4'].fit(Xtr, ytr)
    
    print("  [5/5] RandomForest (400 trees)...")
    models['S5'] = RandomForestRegressor(**STREAM5_RF, random_state=42)
    models['S5'].fit(Xtr, ytr)
    
    use_scaled = {'S3'}
    tp = {}
    for n, m in models.items():
        X = Xte_s if n in use_scaled else Xte
        tp[n] = m.predict(X)
        print(f"    {n}: RMSE={np.sqrt(mean_squared_error(yte, tp[n])):.4f}")
    
    # --- OPTIMIZE FUSION WEIGHTS ---
    print("\n  Optimizing fusion weights (Nelder-Mead, 50 restarts)...")
    val_sz = int(len(Xtr) * FUSION_VALIDATION_FRACTION)
    Xtr2, Xval = Xtr[:-val_sz], Xtr[-val_sz:]
    Xtr2_s = sc.fit_transform(Xtr2); Xval_s = sc.transform(Xval)
    ytr2, yval = ytr[:-val_sz], ytr[-val_sz:]
    
    from sklearn.base import clone
    val_preds = {}
    for n, m in models.items():
        mc = clone(m); mc.fit(Xtr2_s if n in use_scaled else Xtr2, ytr2)
        val_preds[n] = mc.predict(Xval_s if n in use_scaled else Xval)
    
    vm = np.column_stack([val_preds[n] for n in models])
    tm = np.column_stack([tp[n] for n in models])
    
    def obj(w):
        w = np.abs(w)/np.abs(w).sum()
        return np.sqrt(mean_squared_error(yval, vm @ w))
    
    best_w, best_r = None, 1e9
    for _ in range(FUSION_RESTARTS):
        w0 = np.random.dirichlet(np.ones(len(models)))
        res = optimize.minimize(obj, w0, method=FUSION_OPTIMIZER, options={'maxiter':2000})
        if res.fun < best_r: best_r = res.fun; best_w = np.abs(res.x)/np.abs(res.x).sum()
    
    Xtr_s = sc.fit_transform(Xtr); Xte_s = sc.transform(Xte)
    blend = tm @ best_w
    print(f"  Weights: {dict(zip(models.keys(), [f'{w:.3f}' for w in best_w]))}")
    print(f"  Blend RMSE: {np.sqrt(mean_squared_error(yte, blend)):.4f}")
    
    # --- RESIDUAL CALIBRATION ---
    print("  Residual calibration...")
    val_blend = vm @ best_w
    val_resid = yval - val_blend
    rfeat_val = np.column_stack([val_blend, val_blend**2,
        TR.iloc[-val_sz:]['sin1'].values, TR.iloc[-val_sz:]['cos1'].values,
        TR.iloc[-val_sz:]['Rs7'].values, TR.iloc[-val_sz:]['Anom'].values])
    rfeat_te = np.column_stack([blend, blend**2,
        TE['sin1'].values, TE['cos1'].values, TE['Rs7'].values, TE['Anom'].values])
    
    rcal = HistGradientBoostingRegressor(**RESIDUAL_CALIBRATOR, random_state=42)
    rcal.fit(rfeat_val, val_resid)
    cal_pred = blend + rcal.predict(rfeat_te)
    bias = np.mean(val_blend + rcal.predict(rfeat_val) - yval)
    cal_pred -= bias * 0.5
    
    # --- TAIL SPECIALISTS ---
    print("  Tail specialists...")
    hw = np.ones(len(ytr)); hw[ytr >= np.percentile(ytr, TAIL_HOT_PERCENTILE)] = TAIL_WEIGHT
    hs = HistGradientBoostingRegressor(**TAIL_SPECIALIST, random_state=42)
    hs.fit(Xtr, ytr, sample_weight=hw); hsp = hs.predict(Xte)
    
    cw = np.ones(len(ytr)); cw[ytr <= np.percentile(ytr, TAIL_COLD_PERCENTILE)] = TAIL_WEIGHT
    cs = HistGradientBoostingRegressor(**TAIL_SPECIALIST, random_state=42)
    cs.fit(Xtr, ytr, sample_weight=cw); csp = cs.predict(Xte)
    
    p90t, p95t = np.percentile(ytr, 90), np.percentile(ytr, 95)
    p10t, p5t = np.percentile(ytr, 10), np.percentile(ytr, 5)
    hb = np.clip((cal_pred-p90t)/(p95t-p90t+0.1), 0, TAIL_BLEND_MAX)
    cb = np.clip((p10t-cal_pred)/(p10t-p5t+0.1), 0, TAIL_BLEND_MAX)
    final = (1-hb-cb)*cal_pred + hb*hsp + cb*csp
    
    print(f"  Final RMSE: {np.sqrt(mean_squared_error(yte, final)):.4f}")
    
    # Update results
    prev['predictions']['mmwstm_adran'] = [round(float(v),2) for v in final]
    prev['blend_weights'] = dict(zip(models.keys(), [round(float(w),4) for w in best_w]))
    with open(RESULTS_FILE, 'w') as f: json.dump(prev, f, indent=2, default=str)
    print(f"  Updated {RESULTS_FILE}")

if __name__ == "__main__":
    main()
