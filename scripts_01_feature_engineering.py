#!/usr/bin/env python3
"""Step 1: Feature engineering from raw Baghdad meteorological data."""
import numpy as np, pandas as pd, os, sys, warnings
warnings.filterwarnings('ignore')
from scipy import stats

sys.path.insert(0, os.path.dirname(__file__))
from config import *

def main():
    os.makedirs(RESULTS_DIR, exist_ok=True)
    print("Loading raw data...")
    df = pd.read_excel(RAW_DATA_FILE)
    df = df.dropna(subset=[DATE_COL])
    df[DATE_COL] = pd.to_datetime(df[DATE_COL])
    df = df.sort_values(DATE_COL).reset_index(drop=True)
    print(f"  Raw: {len(df)} rows, {df.shape[1]} columns")
    print(f"  Range: {df[DATE_COL].min().date()} to {df[DATE_COL].max().date()}")
    print(f"  Tmax: {df[TARGET].min():.1f} to {df[TARGET].max():.1f} °C")

    # Clean numeric columns
    num_cols = [c for c in df.columns if df[c].dtype in ['float64','int64'] or c in TODAY_VARS]
    for c in num_cols:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors='coerce')
            df[c] = df[c].fillna(df[c].rolling(7, min_periods=1, center=True).mean()).fillna(df[c].mean())

    # --- TEMPORAL ---
    df['doy'] = df[DATE_COL].dt.dayofyear
    df['month'] = df[DATE_COL].dt.month
    for k in SEASONAL_HARMONICS:
        df[f'sin{k}'] = np.sin(2*k*np.pi*df['doy']/365.25)
        df[f'cos{k}'] = np.cos(2*k*np.pi*df['doy']/365.25)

    # --- TODAY'S FEATURES ---
    for c in TODAY_VARS:
        if c in df.columns: df[f'{c}_t0'] = df[c]

    # --- LAGS ---
    for lag in LAG_DAYS:
        for c in LAG_VARS:
            if c in df.columns: df[f'{c}_L{lag}'] = df[c].shift(lag)

    # --- ROLLING STATS ---
    for w in ROLLING_WINDOWS:
        r = df[TARGET].rolling(w, min_periods=1)
        df[f'Rm{w}'] = r.mean(); df[f'Rs{w}'] = r.std().fillna(0)
        df[f'Rn{w}'] = r.min(); df[f'Rx{w}'] = r.max()
        df[f'Rr{w}'] = df[f'Rx{w}'] - df[f'Rn{w}']
        df[f'Rsk{w}'] = r.skew().fillna(0)
    for w in ROLLING_DETAIL_WINDOWS:
        for c2 in ROLLING_DETAIL_VARS:
            if c2 in df.columns:
                df[f'{c2}_Rm{w}'] = df[c2].rolling(w, min_periods=1).mean()
                df[f'{c2}_Rs{w}'] = df[c2].rolling(w, min_periods=1).std().fillna(0)

    # --- TRENDS ---
    for w in TREND_WINDOWS:
        v = df[TARGET].values; s = np.full(len(v), np.nan)
        for i in range(w-1, len(v)):
            y = v[i-w+1:i+1]
            if not np.any(np.isnan(y)): s[i] = stats.linregress(np.arange(w), y).slope
        df[f'Tr{w}'] = s

    # --- ANOMALIES ---
    dm = df.groupby('doy')[TARGET].transform(lambda x: x.shift(1).expanding().mean())
    ds = df.groupby('doy')[TARGET].transform(lambda x: x.shift(1).expanding().std()).fillna(1)
    df['Anom'] = ((df[TARGET]-dm)/ds).fillna(0).clip(-5,5)
    df['AnomAbs'] = df['Anom'].abs()
    for w in [7,14]: df[f'RA{w}'] = df[TARGET] - df[f'Rm{w}']

    # --- INTERACTIONS ---
    df['HxT'] = df['humidity']*df[TARGET]
    df['SxC'] = df['solarradiation']*(100-df['cloudcover'])
    df['WxP'] = df['windspeed']*df['sealevelpressure']
    df['Ddep'] = df[TARGET]-df['dew']
    df['Trng'] = df[TARGET]-df['tempmin']
    df['Fdiff'] = df['feelslikemax']-df[TARGET]
    df['THI'] = df[TARGET]*df['humidity']/100
    df['SWR'] = df['solarradiation']/(df['windspeed']+1)

    # --- CHANGES ---
    for c in CHANGE_VARS:
        if c in df.columns:
            df[f'{c}_d1'] = df[c]-df[c].shift(1)
            df[f'{c}_d3'] = df[c]-df[c].shift(3)

    # --- EXTREMES ---
    df['isH'] = (df[TARGET]>HOT_FLAG_THRESHOLD).astype(float)
    df['isC'] = (df[TARGET]<COLD_FLAG_THRESHOLD).astype(float)
    df['conH'] = 0.0
    for i in range(1, len(df)):
        if df.iloc[i][TARGET] > HOT_FLAG_THRESHOLD:
            df.loc[df.index[i], 'conH'] = df.iloc[i-1]['conH'] + 1

    # --- EWM ---
    for sp in EWM_SPANS: df[f'E{sp}'] = df[TARGET].ewm(span=sp).mean()

    # --- TARGET ---
    df['target_next'] = df[TARGET].shift(-1)

    # --- FINALIZE ---
    exclude = {DATE_COL,'target_next','conditions','description','icon',
               'sunrise','sunset','preciptype','snow','snowdepth',TARGET}
    feat_cols = [c for c in df.columns if c not in exclude
                 and df[c].dtype in ['float64','int64','float32','int32','uint32']]

    dfm = df.dropna(subset=['target_next']).iloc[MIN_LAG_DAYS:].reset_index(drop=True)
    for c in feat_cols: dfm[c] = dfm[c].fillna(dfm[c].median())

    # Save
    save_cols = [DATE_COL, TARGET, 'target_next', 'doy', 'month'] + feat_cols
    dfm[save_cols].to_csv(FEATURES_FILE, index=False)

    print(f"\n  Engineered: {len(dfm)} rows, {len(feat_cols)} features")
    print(f"  Saved to: {FEATURES_FILE}")
    print(f"  Feature groups: thermal={sum(1 for c in feat_cols if 'temp' in c.lower() or 'feels' in c.lower() or '_L' in c)}, "
          f"moisture={sum(1 for c in feat_cols if 'dew' in c or 'humid' in c or 'precip' in c)}, "
          f"rolling={sum(1 for c in feat_cols if c.startswith('R') and c[1] in 'msnxrk')}, "
          f"anomaly={sum(1 for c in feat_cols if 'Anom' in c or 'Tr' in c[:2] or '_d1' in c or '_d3' in c)}")

if __name__ == "__main__":
    main()
