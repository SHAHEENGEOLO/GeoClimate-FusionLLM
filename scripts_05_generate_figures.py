#!/usr/bin/env python3
"""Step 5: Generate all publication figures."""
import numpy as np, pandas as pd, json, os, sys, warnings
warnings.filterwarnings('ignore')
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from sklearn.metrics import mean_squared_error, r2_score
sys.path.insert(0, os.path.dirname(__file__))
from config import *

def main():
    os.makedirs(FIGURES_DIR, exist_ok=True)
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10,'figure.dpi':FIGURE_DPI,
                         'savefig.dpi':FIGURE_DPI,'savefig.bbox':'tight','figure.facecolor':'white'})
    
    with open(RESULTS_FILE) as f: D = json.load(f)
    yte = np.array(D['predictions']['y_test'])
    dates = pd.to_datetime(D['predictions']['dates'])
    df = pd.read_excel(RAW_DATA_FILE)
    df = df.dropna(subset=['datetime']); df['datetime'] = pd.to_datetime(df['datetime'])
    df = df.sort_values('datetime').reset_index(drop=True)
    df['month'] = df['datetime'].dt.month; df['year'] = df['datetime'].dt.year
    
    def save(name): plt.savefig(os.path.join(FIGURES_DIR, f'{name}.{FIGURE_FORMAT}')); plt.close()
    
    # F1: Time series
    fig, ax = plt.subplots(figsize=(14,5))
    ax.plot(df['datetime'], df['tempmax'], color='#fc8d62', alpha=0.4, lw=0.5, label='Daily Tmax')
    ax.plot(df['datetime'], df['tempmax'].rolling(30,center=True).mean(), color='#1b9e77', lw=2, label='30-d mean')
    ax.axvline(x=pd.Timestamp(D['test_dates'][0]), color='red', ls='--', lw=1.5, label='Train/Test split')
    ax.set_ylabel('Tmax (°C)'); ax.set_title('Baghdad Daily Maximum Temperature (2019-2024)', fontweight='bold')
    ax.legend(fontsize=8); ax.grid(alpha=0.3); save('fig01_timeseries')
    
    # F2: Monthly climatology
    ms = df.groupby('month')['tempmax'].agg(['mean','std'])
    ms['p5'] = df.groupby('month')['tempmax'].quantile(0.05).values
    ms['p95'] = df.groupby('month')['tempmax'].quantile(0.95).values
    fig, ax = plt.subplots(figsize=(10,6))
    ax.fill_between(range(12), ms['p5'], ms['p95'], alpha=0.2, color='#fc8d62', label='5-95th pctl')
    ax.fill_between(range(12), ms['mean']-ms['std'], ms['mean']+ms['std'], alpha=0.3, color='#8da0cb', label='±1σ')
    ax.plot(range(12), ms['mean'], 'ko-', lw=2, ms=6, label='Mean')
    ax.set_xticks(range(12)); ax.set_xticklabels(['J','F','M','A','M','J','J','A','S','O','N','D'])
    ax.set_ylabel('Tmax (°C)'); ax.set_title('Monthly Climatology', fontweight='bold'); ax.legend(); ax.grid(alpha=0.3)
    save('fig02_climatology')
    
    # F3: RMSE bars
    mdls = ['Persist.','Clim.','AR(1)+S','MLR','MLP-D','MLP-S','GBRT','MMWSTM\nADRAN+']
    rmses = [D['baselines'][k]['RMSE'] for k in ['persistence','climatology','ar_seasonal','mlr','mlp_deep','mlp_shallow','gbrt']]
    mm_rmse = np.sqrt(mean_squared_error(yte, np.array(D['predictions']['mmwstm_adran'])))
    rmses.append(mm_rmse)
    colors = ['#969696','#969696','#969696','#fc8d62','#bdbdbd','#bdbdbd','#8da0cb','#e41a1c']
    fig, ax = plt.subplots(figsize=(12,6))
    bars = ax.bar(range(len(mdls)), rmses, color=colors, edgecolor='black', lw=0.8, width=0.7)
    for b,v in zip(bars,rmses): ax.text(b.get_x()+b.get_width()/2,v+0.03,f'{v:.3f}',ha='center',fontsize=9,fontweight='bold')
    ax.set_xticks(range(len(mdls))); ax.set_xticklabels(mdls)
    ax.set_ylabel('RMSE (°C)'); ax.set_title('Overall RMSE Comparison', fontweight='bold')
    ax.set_ylim(0,4); ax.grid(axis='y',alpha=0.3); save('fig03_rmse')
    
    # F4: Scatter
    fig, axes = plt.subplots(1,3, figsize=(16,5))
    for ax, k, nm, cl in [(axes[0],'mlr','Ridge MLR','#fc8d62'),
                            (axes[1],'gbrt','GBRT','#8da0cb'),
                            (axes[2],'mmwstm_adran','MMWSTM-ADRAN+','#e41a1c')]:
        yp = np.array(D['predictions'][k])
        ax.scatter(yte, yp, alpha=0.4, s=15, c=cl, edgecolors='none')
        mn2,mx2 = min(yte.min(),yp.min())-1, max(yte.max(),yp.max())+1
        ax.plot([mn2,mx2],[mn2,mx2],'k--',lw=1,alpha=0.5)
        ax.set_title(f'{nm}\nRMSE={np.sqrt(mean_squared_error(yte,yp)):.3f}°C',fontweight='bold')
        ax.set_xlabel('Observed (°C)'); ax.set_ylabel('Predicted (°C)'); ax.set_aspect('equal'); ax.grid(alpha=0.3)
    plt.tight_layout(); save('fig04_scatter')
    
    # F5: Test time series
    fig, ax = plt.subplots(figsize=(14,6))
    ax.plot(dates, yte, 'k-', lw=1, alpha=0.8, label='Observed')
    ax.plot(dates, D['predictions']['mmwstm_adran'], color='#e41a1c', lw=1, alpha=0.7, label='MMWSTM-ADRAN+')
    ax.plot(dates, D['predictions']['mlr'], color='#fc8d62', lw=0.8, alpha=0.5, label='MLR')
    ax.set_ylabel('Tmax (°C)'); ax.set_title('Test Period Predictions', fontweight='bold')
    ax.legend(fontsize=9); ax.grid(alpha=0.3); save('fig05_test_timeseries')
    
    # F6: Residuals
    fig, axes = plt.subplots(1,3, figsize=(15,5))
    for ax, k, nm, cl in [(axes[0],'mlr','MLR','#fc8d62'),(axes[1],'gbrt','GBRT','#8da0cb'),
                            (axes[2],'mmwstm_adran','MMWSTM-ADRAN+','#e41a1c')]:
        res = yte - np.array(D['predictions'][k])
        ax.hist(res, bins=40, color=cl, alpha=0.7, edgecolor='black', lw=0.5, density=True)
        ax.axvline(x=0, color='black', ls='--'); ax.axvline(x=np.mean(res), color='red', lw=1.5)
        ax.set_title(f'{nm}\nBias={np.mean(res):.3f}, σ={np.std(res):.3f}°C', fontweight='bold')
        ax.set_xlabel('Residual (°C)'); ax.grid(alpha=0.3)
    plt.tight_layout(); save('fig06_residuals')
    
    n_figs = len([f for f in os.listdir(FIGURES_DIR) if f.endswith(f'.{FIGURE_FORMAT}')])
    print(f"  Generated {n_figs} figures in {FIGURES_DIR}/")

if __name__ == "__main__":
    main()
