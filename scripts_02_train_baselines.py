#!/usr/bin/env python3
"""Step 2: Train classical and neural baselines."""
import numpy as np, pandas as pd, os, sys, json, warnings, pickle
warnings.filterwarnings('ignore')
sys.path.insert(0, os.path.dirname(__file__))
from config import *
from sklearn.preprocessing import RobustScaler
from sklearn.linear_model import LinearRegression, RidgeCV
from sklearn.neural_network import MLPRegressor
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

def main():
    print("Loading engineered features...")
    df = pd.read_csv(FEATURES_FILE, parse_dates=[DATE_COL])
    feat_cols = [c for c in df.columns if c not in {DATE_COL, TARGET, 'target_next', 'doy', 'month'}]
    
    si = int(len(df) * TRAIN_SPLIT_RATIO)
    TR, TE = df.iloc[:si], df.iloc[si:]
    Xtr, ytr = TR[feat_cols].values.astype(np.float32), TR['target_next'].values.astype(np.float32)
    Xte, yte = TE[feat_cols].values.astype(np.float32), TE['target_next'].values.astype(np.float32)
    
    sc = RobustScaler(); Xtr_s = sc.fit_transform(Xtr); Xte_s = sc.transform(Xte)
    P95 = np.percentile(yte, TEST_HOT_PERCENTILE)
    P5 = np.percentile(yte, TEST_COLD_PERCENTILE)
    hm, cm = yte >= P95, yte <= P5
    
    def ev(yt, yp, n=""):
        return {'name':n, 'RMSE':float(np.sqrt(mean_squared_error(yt,yp))),
                'MAE':float(mean_absolute_error(yt,yp)), 'R2':float(r2_score(yt,yp)),
                'Corr':float(np.corrcoef(yt,yp)[0,1]), 'Bias':float(np.mean(yp-yt)),
                'Hot_RMSE':float(np.sqrt(mean_squared_error(yt[hm],yp[hm]))),
                'Cold_RMSE':float(np.sqrt(mean_squared_error(yt[cm],yp[cm])))}
    
    results = {}
    
    # Persistence
    pp = TE[f'{TARGET}_t0'].values if f'{TARGET}_t0' in TE.columns else TE[TARGET].values
    results['persistence'] = ev(yte, pp, "Persistence")
    
    # Seasonal climatology
    tdm = TR.groupby('doy')['target_next'].mean().to_dict()
    cp = TE['doy'].map(tdm).fillna(TR['target_next'].mean()).values
    results['climatology'] = ev(yte, cp, "Seasonal Climatology")
    
    # AR(1)+seasonal
    aXtr = TR[['tempmax_t0','sin1','cos1']].values
    aXte = TE[['tempmax_t0','sin1','cos1']].values
    arp = LinearRegression().fit(aXtr, ytr).predict(aXte)
    results['ar_seasonal'] = ev(yte, arp, "AR(1)+Seasonal")
    
    # Ridge MLR
    ridge = RidgeCV(alphas=np.logspace(-2,3,30), cv=TimeSeriesSplit(RIDGE_CV_FOLDS))
    ridge.fit(Xtr_s, ytr); mlrp = ridge.predict(Xte_s)
    results['mlr'] = ev(yte, mlrp, "Ridge MLR")
    
    # MLP-Deep
    mlp1 = MLPRegressor(**MLP_DEEP, random_state=42)
    mlp1.fit(Xtr_s, ytr); mlp1p = mlp1.predict(Xte_s)
    results['mlp_deep'] = ev(yte, mlp1p, "MLP-Deep (256-128-64-32)")
    
    # MLP-Shallow
    mlp2 = MLPRegressor(**MLP_SHALLOW, random_state=42)
    mlp2.fit(Xtr_s, ytr); mlp2p = mlp2.predict(Xte_s)
    results['mlp_shallow'] = ev(yte, mlp2p, "MLP-Shallow (128-128-64)")
    
    # GBRT
    gbrt = GradientBoostingRegressor(**GBRT_BASELINE, random_state=42)
    gbrt.fit(Xtr, ytr); gbrtp = gbrt.predict(Xte)
    results['gbrt'] = ev(yte, gbrtp, "GBRT (400 trees, depth 5)")
    
    # Print results
    pr = results['persistence']['RMSE']
    print(f"\n{'Model':42s} {'RMSE':>7s} {'MAE':>7s} {'R²':>6s} {'Hot':>7s} {'Cold':>7s} {'Skill':>6s}")
    print("-"*85)
    for r in results.values():
        sk = (1-r['RMSE']/pr)*100
        print(f"  {r['name']:40s} {r['RMSE']:7.4f} {r['MAE']:7.4f} {r['R2']:6.4f} {r['Hot_RMSE']:7.4f} {r['Cold_RMSE']:7.4f} {sk:+6.1f}%")
    
    # Save
    save = {'baselines': results, 'persist_rmse': pr, 'hot_thr': float(P95), 'cold_thr': float(P5),
            'n_train': len(TR), 'n_test': len(TE), 'n_features': len(feat_cols),
            'train_dates': [str(TR[DATE_COL].min().date()), str(TR[DATE_COL].max().date())],
            'test_dates': [str(TE[DATE_COL].min().date()), str(TE[DATE_COL].max().date())],
            'predictions': {'dates': [str(d.date()) for d in TE[DATE_COL]],
                           'y_test': yte.tolist(), 'persist': pp.tolist(),
                           'mlr': mlrp.tolist(), 'ar': arp.tolist(), 'gbrt': gbrtp.tolist()},
            'ridge_alpha': float(ridge.alpha_), 'feature_cols': feat_cols}
    
    with open(RESULTS_FILE, 'w') as f: json.dump(save, f, indent=2, default=str)
    print(f"\nSaved to {RESULTS_FILE}")

if __name__ == "__main__":
    main()
