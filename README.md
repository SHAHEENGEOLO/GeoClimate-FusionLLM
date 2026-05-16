# GeoClimate-FusionLLM

**Multi-Modal Data Fusion and LLM-Assisted Explainable AI for Environmental Early-Warning in Arid Urban Regions**

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![DOI](https://img.shields.io/badge/DOI-pending-lightgrey.svg)]()

DOI: 10.5281/zenodo.20243916
---

## Overview

GeoClimate-FusionLLM is a reproducible framework for short-range environmental early warning in arid urban regions. It combines:

- **Multi-modal feature engineering** — 159 features from 7 physical modality groups
- **MMWSTM-ADRAN+** — A multi-stream ensemble with extreme-aware tail calibration
- **Transparent benchmarking** — Classical, neural, and tree-ensemble baselines with Diebold-Mariano significance tests
- **LLM/RAG explanation layer** — Evidence-constrained warning narratives (conceptual design)

### Key Results (Baghdad, Iraq — 2019–2024)

| Model | RMSE (°C) | Skill vs Persistence | Cold-tail RMSE | DM p-value vs MLR |
|-------|-----------|---------------------|----------------|-------------------|
| Persistence | 2.240 | — | 3.393 | — |
| Ridge MLR | 2.036 | +9.1% | 3.312 | — |
| GBRT (baseline) | 1.965 | +12.2% | 3.224 | 0.126 (n.s.) |
| **MMWSTM-ADRAN+** | **1.973** | **+11.9%** | **3.079** | **0.009 (\*\*)** |

> MMWSTM-ADRAN+ is the **only model that significantly outperforms MLR** (p = 0.009) while achieving the **best cold-tail RMSE** among all models.

---

## Repository Structure

```
GeoClimate-FusionLLM/
├── README.md                          # This file
├── LICENSE                            # MIT License
├── requirements.txt                   # Python dependencies
├── CITATION.cff                       # Citation metadata
├── .gitignore                         # Git ignore rules
│
├── data/
│   └── baghdad_2019-2024.xlsx         # Raw daily meteorological data
│
├── scripts/
│   ├── run_all.py                     # Master pipeline (runs everything)
│   ├── 01_feature_engineering.py      # Raw data → 159 engineered features
│   ├── 02_train_baselines.py          # Classical + neural baselines
│   ├── 03_train_mmwstm_adran.py       # MMWSTM-ADRAN+ multi-stream ensemble
│   ├── 04_evaluate.py                 # Metrics, DM tests, ablation
│   ├── 05_generate_figures.py         # All 15 publication figures
│   └── config.py                      # Hyperparameters and paths
│
├── results/
│   └── paper_results.json             # All model predictions and metrics
│
├── figures/                           # Generated publication figures (PNG)
│
├── paper/
│   └── GeoClimate_FusionLLM_R2.docx  # Manuscript (latest revision)
│
└── docs/
    └── METHODOLOGY.md                 # Detailed methodology notes
```

---

## Quick Start

### 1. Clone and install

```bash
git clone https://github.com/YOUR_USERNAME/GeoClimate-FusionLLM.git
cd GeoClimate-FusionLLM
pip install -r requirements.txt
```

### 2. Run the full pipeline

```bash
python scripts/run_all.py
```

This will:
1. Engineer 159 features from raw Baghdad data
2. Train all baseline models (persistence, climatology, AR, MLR, GBRT, MLPs)
3. Train MMWSTM-ADRAN+ multi-stream ensemble with tail calibration
4. Evaluate all models with Diebold-Mariano significance tests
5. Run 3-seed robustness analysis
6. Run ablation diagnostics
7. Generate all 15 publication figures

### 3. Run individual scripts

```bash
python scripts/01_feature_engineering.py   # Feature engineering only
python scripts/02_train_baselines.py       # Baselines only
python scripts/03_train_mmwstm_adran.py    # MMWSTM-ADRAN+ only
python scripts/04_evaluate.py              # Evaluation + statistical tests
python scripts/05_generate_figures.py      # Figures only
```

---

## Data

**Source:** [Visual Crossing Weather API](https://www.visualcrossing.com)  
**Location:** Baghdad, Iraq (33.31°N, 44.37°E)  
**Period:** 2019-01-01 to 2024-12-31  
**Records:** 2,192 daily observations  
**Variables:** 30 original columns (thermal, moisture, wind, pressure, radiation, cloud, astronomical)

### Descriptive Statistics (Tmax)

| Statistic | Value |
|-----------|-------|
| Minimum | 6.1 °C |
| Maximum | 51.1 °C |
| Mean | 31.87 °C |
| Std Dev | 10.73 °C |
| P5 / P50 / P95 | 16.0 / 32.0 / 47.0 °C |

---

## Feature Engineering (159 Features)

| Modality Group | Count | Description |
|---------------|-------|-------------|
| Thermal | 35 | Same-day + lagged Tmax, Tmin, Tmean, feels-like |
| Moisture | 20 | Dew point, humidity, precipitation + lags + rolling |
| Dynamic | 15 | Wind speed/direction, pressure + lags |
| Radiation | 12 | Solar radiation, UV, cloud cover + rolling |
| Temporal | 8 | Harmonic seasonal encodings (k=1,2,3), month |
| Rolling-statistical | 40 | Multi-scale (3–30d) mean, σ, min, max, range, skew |
| Anomaly & change | 29 | DOY-anomaly, rolling anomaly, trends, change features |

---

## MMWSTM-ADRAN+ Architecture

```
                    ┌─────────────────────────────────┐
                    │   Multi-Modal Input (159 feat)   │
                    └───────────────┬─────────────────┘
                                    │
          ┌─────────┬───────┬───────┼───────┬─────────┐
          ▼         ▼       ▼       ▼       ▼         │
     ┌─────────┐ ┌──────┐ ┌─────┐ ┌──────┐ ┌──────┐  │
     │HistGBRT │ │HistG │ │ DNN │ │Extra │ │  RF  │  │
     │depth=6  │ │BRT   │ │512→ │ │Trees │ │400   │  │
     │Stream 1 │ │d=4   │ │64   │ │400   │ │trees │  │
     │(MMWSTM) │ │Str.2 │ │Str.3│ │Str.4 │ │Str.5 │  │
     └────┬────┘ └──┬───┘ └──┬──┘ └──┬───┘ └──┬───┘  │
          └─────────┴────────┼───────┴────────┘       │
                             ▼                        │
                ┌────────────────────────┐            │
                │  Optimized Weighted    │            │
                │  Fusion (Nelder-Mead)  │            │
                └───────────┬────────────┘            │
                            ▼                         │
                ┌────────────────────────┐            │
                │  Residual Calibration  │            │
                └───────────┬────────────┘            │
                            ▼                         │
                ┌────────────────────────┐            │
                │  Tail-Specialist       │            │
                │  Blending (hot/cold)   │            │
                └───────────┬────────────┘            │
                            ▼                         │
                ┌────────────────────────┐   ┌────────┴───────┐
                │   Final Prediction     │──▶│  LLM/RAG       │
                │   ŷ_{t+1}             │   │  Explanation    │
                └────────────────────────┘   └────────────────┘
```

---

## Evaluation Metrics

- **RMSE** — Root Mean Square Error (°C)
- **MAE** — Mean Absolute Error (°C)
- **R²** — Coefficient of Determination
- **Skill(%)** — RMSE improvement over persistence
- **Hot-tail RMSE** — RMSE on days ≥ P95 (46.5 °C)
- **Cold-tail RMSE** — RMSE on days ≤ P5 (17.0 °C)
- **DM test** — Diebold-Mariano test for statistical significance

---

## Citation

If you use this code or data in your research, please cite:

```bibtex
@article{ahmed2025geoclimate,
  title={GeoClimate-FusionLLM: Multi-Modal Data Fusion and LLM-Assisted 
         Explainable AI for Environmental Early-Warning in Arid Urban Regions},
  author={Ahmed, Shaheen Mohammed Saleh and G{\"u}neyli, Hakan},
  journal={[Journal Name]},
  year={2025},
  note={Under review}
}
```

---

## License

This project is licensed under the MIT License — see [LICENSE](LICENSE) for details.

---

## Acknowledgments

- Weather data provided by [Visual Crossing](https://www.visualcrossing.com)
- Built with [scikit-learn](https://scikit-learn.org), [pandas](https://pandas.pydata.org), [matplotlib](https://matplotlib.org), [scipy](https://scipy.org)
