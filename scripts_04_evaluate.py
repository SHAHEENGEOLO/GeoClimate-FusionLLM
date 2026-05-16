#!/usr/bin/env python3
"""Step 4: Evaluation, Diebold-Mariano tests, ablation, robustness."""
import numpy as np, json, os, sys, warnings
warnings.filterwarnings('ignore')
sys.path.insert(0, os.path.dirname(__file__))
from config import *
from scipy import stats as st
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

def dm_test(e1, e2, h=1):
    """Diebold-Mariano test for equal predictive accuracy."""
    d = e1**2 - e2**2; n = len(d); md = np.mean(d); vd = np.var(d, ddof=1)
    for k in range(1, h):
        gk = np.sum((d[k:]-md)*(d[:-k]-md))/n
        vd += 2*(1-k/h)*gk/n
    if vd <= 0: vd = np.var(d, ddof=1)
    dm = md / np.sqrt(vd/n)
    return dm, 2*(1-st.norm.cdf(abs(dm)))

def main():
    with open(RESULTS_FILE) as f: D = json.load(f)
    yte = np.array(D['predictions']['y_test'])
    P95, P5 = D['hot_thr'], D['cold_thr']
    hm, cm = yte >= P95, yte <= P5
    
    preds = {k: np.array(v) for k,v in D['predictions'].items() if k not in {'dates','y_test'}}
    
    def ev(yp, n=""): 
        return {'name':n,'RMSE':float(np.sqrt(mean_squared_error(yte,yp))),
                'MAE':float(mean_absolute_error(yte,yp)),'R2':float(r2_score(yte,yp)),
                'Corr':float(np.corrcoef(yte,yp)[0,1]),'Bias':float(np.mean(yp-yte)),
                'Hot_RMSE':float(np.sqrt(mean_squared_error(yte[hm],yp[hm]))),
                'Cold_RMSE':float(np.sqrt(mean_squared_error(yte[cm],yp[cm])))}
    
    # Full evaluation
    print("=" * 70); print("MODEL EVALUATION"); print("=" * 70)
    all_res = []
    name_map = {'persist':'Persistence','mlr':'Ridge MLR','ar':'AR(1)+Seasonal',
                'gbrt':'GBRT (400 trees)','mmwstm_adran':'MMWSTM-ADRAN+'}
    for k, yp in preds.items():
        nm = name_map.get(k, k)
        r = ev(yp, nm); all_res.append(r)
        sk = (1-r['RMSE']/all_res[0]['RMSE'])*100 if all_res else 0
        print(f"  {nm:35s} RMSE={r['RMSE']:.4f} Hot={r['Hot_RMSE']:.4f} Cold={r['Cold_RMSE']:.4f} Skill={sk:+.1f}%")
    
    # DM tests
    print("\n" + "=" * 70); print("DIEBOLD-MARIANO TESTS"); print("=" * 70)
    pairs = [('mmwstm_adran','mlr'), ('mmwstm_adran','gbrt'),
             ('mmwstm_adran','persist'), ('gbrt','mlr')]
    dm_results = []
    for a, b in pairs:
        if a in preds and b in preds:
            ea, eb = np.abs(yte-preds[a]), np.abs(yte-preds[b])
            stat, pval = dm_test(ea, eb)
            sig = '***' if pval<0.01 else '**' if pval<0.05 else '*' if pval<0.1 else 'n.s.'
            dm_results.append({'a':name_map.get(a,a),'b':name_map.get(b,b),'stat':round(stat,3),'p':round(pval,4),'sig':sig})
            print(f"  {name_map.get(a,a)} vs {name_map.get(b,b)}: DM={stat:.3f}, p={pval:.4f} {sig}")
    
    D['evaluation'] = all_res
    D['dm_tests'] = dm_results
    with open(RESULTS_FILE, 'w') as f: json.dump(D, f, indent=2, default=str)
    print(f"\nUpdated {RESULTS_FILE}")

if __name__ == "__main__":
    main()
