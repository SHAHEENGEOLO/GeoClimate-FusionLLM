from __future__ import annotations

from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from .evaluation import regression_metrics


def run_simple_baselines(
    train: pd.DataFrame,
    val: pd.DataFrame,
    test: pd.DataFrame,
    feature_cols: List[str],
    train_high_threshold: float,
    train_low_threshold: float,
) -> Tuple[pd.DataFrame, Dict[str, np.ndarray]]:
    """Run reviewer-requested simple baselines on the same chronological split."""
    results: Dict[str, Dict[str, float]] = {}
    predictions: Dict[str, np.ndarray] = {}

    y_test = test["target"].values

    # 1. Persistence: y_hat(t+1)=tempmax(t)
    pred = test["persistence"].values
    predictions["Persistence"] = pred
    results["Persistence"] = regression_metrics(y_test, pred, train_high_threshold, train_low_threshold)

    # 2. Seasonal climatology from training target day-of-year.
    train_clim = train.copy()
    train_clim["target_dayofyear"] = pd.to_datetime(train_clim["target_date"]).dt.dayofyear
    clim = train_clim.groupby("target_dayofyear")["target"].mean().to_dict()
    global_mean = float(train_clim["target"].mean())
    test_doy = pd.to_datetime(test["target_date"]).dt.dayofyear
    pred = np.asarray([clim.get(int(d), global_mean) for d in test_doy], dtype=float)
    predictions["Seasonal climatology"] = pred
    results["Seasonal climatology"] = regression_metrics(y_test, pred, train_high_threshold, train_low_threshold)

    # 3. AR(1)+seasonal regression using tempmax(t) plus cyclical predictors.
    ar_features = [c for c in ["persistence", "month_sin", "month_cos", "dayofyear_sin", "dayofyear_cos"] if c in train.columns]
    ar_model = Pipeline([("scaler", StandardScaler()), ("reg", LinearRegression())])
    ar_model.fit(train[ar_features], train["target"])
    pred = ar_model.predict(test[ar_features])
    predictions["AR(1)+seasonal"] = pred
    results["AR(1)+seasonal"] = regression_metrics(y_test, pred, train_high_threshold, train_low_threshold)

    # 4. Multiple linear regression using selected lagged meteorological predictors.
    # Ridge is used instead of plain OLS to reduce coefficient instability from correlated features.
    mlr_features = [c for c in feature_cols if c in train.columns]
    mlr_model = Pipeline([("scaler", StandardScaler()), ("reg", Ridge(alpha=1.0))])
    mlr_model.fit(train[mlr_features], train["target"])
    pred = mlr_model.predict(test[mlr_features])
    predictions["Multiple linear regression"] = pred
    results["Multiple linear regression"] = regression_metrics(y_test, pred, train_high_threshold, train_low_threshold)

    table = pd.DataFrame([{ "model": name, **metrics } for name, metrics in results.items()])
    return table, predictions
