from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import warnings
from pandas.errors import PerformanceWarning
warnings.simplefilter("ignore", PerformanceWarning)
from sklearn.preprocessing import RobustScaler, StandardScaler


@dataclass
class SplitConfig:
    """Chronological split used in the manuscript revision."""

    train_end: str = "2022-03-14"
    val_start: str = "2022-03-15"
    val_end: str = "2022-12-31"
    test_start: str = "2023-01-01"


def add_causal_features(df: pd.DataFrame, split: SplitConfig) -> pd.DataFrame:
    """
    Add leakage-controlled features.

    Causal rolling windows are right-aligned. Climatological anomaly features are
    fitted on the training period only and then applied to validation/test.
    """
    df = df.copy().sort_values("datetime").reset_index(drop=True)

    dt = pd.to_datetime(df["datetime"])
    df["year"] = dt.dt.year
    df["month"] = dt.dt.month
    df["day"] = dt.dt.day
    df["dayofyear"] = dt.dt.dayofyear
    df["dayofweek"] = dt.dt.dayofweek
    df["quarter"] = dt.dt.quarter
    df["weekofyear"] = dt.dt.isocalendar().week.astype(int)

    # Cyclical encodings.
    df["month_sin"] = np.sin(2 * np.pi * df["month"] / 12.0)
    df["month_cos"] = np.cos(2 * np.pi * df["month"] / 12.0)
    df["dayofyear_sin"] = np.sin(2 * np.pi * df["dayofyear"] / 365.25)
    df["dayofyear_cos"] = np.cos(2 * np.pi * df["dayofyear"] / 365.25)
    df["dayofweek_sin"] = np.sin(2 * np.pi * df["dayofweek"] / 7.0)
    df["dayofweek_cos"] = np.cos(2 * np.pi * df["dayofweek"] / 7.0)

    temp_cols = [c for c in ["tempmax", "tempmin", "temp", "feelslikemax", "feelslikemin", "feelslike"] if c in df.columns]

    if "tempmax" in df.columns and "tempmin" in df.columns:
        df["temp_range"] = df["tempmax"] - df["tempmin"]
        df["temp_volatility_7d"] = df["temp_range"].rolling(7, min_periods=2).std()
        df["temp_volatility_30d"] = df["temp_range"].rolling(30, min_periods=2).std()

    # Right-aligned rolling and difference features.
    for col in temp_cols:
        for window in (3, 7, 14, 30):
            df[f"{col}_{window}d_mean"] = df[col].rolling(window, min_periods=1).mean()
            df[f"{col}_{window}d_min"] = df[col].rolling(window, min_periods=1).min()
            df[f"{col}_{window}d_max"] = df[col].rolling(window, min_periods=1).max()
            df[f"{col}_{window}d_std"] = df[col].rolling(window, min_periods=2).std()
        df[f"{col}_1d_change"] = df[col].diff()
        df[f"{col}_7d_change"] = df[col] - df[f"{col}_7d_mean"]
        # Trailing smoothing only; avoids centered Savitzky-Golay leakage.
        df[f"{col}_smooth_trailing7"] = df[col].rolling(7, min_periods=1).mean()

    if "humidity" in df.columns and "temp" in df.columns:
        df["heat_index_proxy"] = df["temp"] + 0.05 * df["humidity"]
    if "precip" in df.columns and "temp" in df.columns:
        df["drought_index_proxy"] = df["temp"] - 10.0 * df["precip"].fillna(0)
        df["drought_index_30d"] = df["drought_index_proxy"].rolling(30, min_periods=1).mean()

    # Training-only climatology for anomalies.
    train_mask = df["datetime"] <= pd.Timestamp(split.train_end)
    train_df = df.loc[train_mask]
    for col in [c for c in ["tempmax", "tempmin", "temp"] if c in df.columns]:
        clim = train_df.groupby("dayofyear")[col].agg(["mean", "std"]).rename(columns={"mean": f"{col}_clim_mean", "std": f"{col}_clim_std"})
        df = df.merge(clim, left_on="dayofyear", right_index=True, how="left")
        global_mean = float(train_df[col].mean())
        global_std = float(train_df[col].std()) if float(train_df[col].std()) > 0 else 1.0
        df[f"{col}_clim_mean"] = df[f"{col}_clim_mean"].fillna(global_mean)
        df[f"{col}_clim_std"] = df[f"{col}_clim_std"].replace(0, np.nan).fillna(global_std)
        df[f"{col}_anomaly_trainclim"] = df[col] - df[f"{col}_clim_mean"]
        df[f"{col}_zscore_trainclim"] = df[f"{col}_anomaly_trainclim"] / df[f"{col}_clim_std"]

    # Fill numeric missing values causally where possible.
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    df[numeric_cols] = df[numeric_cols].replace([np.inf, -np.inf], np.nan)
    df[numeric_cols] = df[numeric_cols].ffill().bfill()

    return df


def choose_feature_columns(df: pd.DataFrame, split: SplitConfig, target_col: str = "tempmax", max_features: int = 40) -> List[str]:
    """Choose features using training data only."""
    excluded = {"year", "day", "weekofyear"}
    numeric_cols = list(df.select_dtypes(include=[np.number]).columns)
    candidate_cols = [c for c in numeric_cols if c not in excluded]

    # Keep target as an input feature because the problem is explicitly one-step autoregressive.
    train_mask = df["datetime"] <= pd.Timestamp(split.train_end)
    train_cols = list(dict.fromkeys(candidate_cols + [target_col]))
    train = df.loc[train_mask, train_cols].dropna()

    corrs: Dict[str, float] = {}
    y = train[target_col]
    for col in candidate_cols:
        if train[col].nunique() <= 1:
            continue
        corrs[col] = abs(float(train[col].corr(y))) if train[col].corr(y) == train[col].corr(y) else 0.0

    ranked = sorted(corrs, key=corrs.get, reverse=True)
    # Ensure physically important predictors are retained.
    must_have = [c for c in ["tempmax", "tempmin", "temp", "humidity", "sealevelpressure", "windspeed", "cloudcover", "precip", "month_sin", "month_cos", "dayofyear_sin", "dayofyear_cos"] if c in candidate_cols]
    ordered = []
    for c in must_have + ranked:
        if c not in ordered:
            ordered.append(c)
    return ordered[:max_features]


def make_supervised_table(df: pd.DataFrame, feature_cols: List[str], target_col: str = "tempmax") -> pd.DataFrame:
    """Create one-day-ahead supervised rows: predictors at day t, target at day t+1."""
    cols = list(dict.fromkeys(["datetime"] + feature_cols + [target_col]))
    out = df[cols].copy()
    out["issue_date"] = out["datetime"]
    out["target_date"] = out["datetime"].shift(-1)
    out["target"] = out[target_col].shift(-1)
    out["persistence"] = out[target_col]
    return out.dropna(subset=["target", "target_date"]).reset_index(drop=True)


def split_supervised(supervised: pd.DataFrame, split: SplitConfig) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Split supervised table by target date."""
    td = pd.to_datetime(supervised["target_date"])
    train = supervised[td <= pd.Timestamp(split.train_end)].copy()
    val = supervised[(td >= pd.Timestamp(split.val_start)) & (td <= pd.Timestamp(split.val_end))].copy()
    test = supervised[td >= pd.Timestamp(split.test_start)].copy()
    return train, val, test


def create_sequence_arrays(
    df: pd.DataFrame,
    feature_cols: List[str],
    split: SplitConfig,
    target_col: str = "tempmax",
    sequence_length: int = 30,
) -> Dict[str, object]:
    """Create scaled sequence arrays for neural models."""
    df = df.sort_values("datetime").reset_index(drop=True).copy()

    samples_x, samples_y, issue_dates, target_dates = [], [], [], []
    for end_idx in range(sequence_length - 1, len(df) - 1):
        start_idx = end_idx - sequence_length + 1
        samples_x.append(df.loc[start_idx:end_idx, feature_cols].values)
        samples_y.append(df.loc[end_idx + 1, target_col])
        issue_dates.append(df.loc[end_idx, "datetime"])
        target_dates.append(df.loc[end_idx + 1, "datetime"])

    X = np.asarray(samples_x, dtype=np.float32)
    y = np.asarray(samples_y, dtype=np.float32).reshape(-1, 1)
    meta = pd.DataFrame({"issue_date": issue_dates, "target_date": target_dates})
    td = pd.to_datetime(meta["target_date"])
    train_idx = np.where(td <= pd.Timestamp(split.train_end))[0]
    val_idx = np.where((td >= pd.Timestamp(split.val_start)) & (td <= pd.Timestamp(split.val_end)))[0]
    test_idx = np.where(td >= pd.Timestamp(split.test_start))[0]

    # Fit scalers using training sequence values only.
    scaler_x = RobustScaler()
    x_train_flat = X[train_idx].reshape(-1, X.shape[-1])
    scaler_x.fit(x_train_flat)

    scaler_y = StandardScaler()
    scaler_y.fit(y[train_idx])

    X_scaled = scaler_x.transform(X.reshape(-1, X.shape[-1])).reshape(X.shape)
    y_scaled = scaler_y.transform(y)

    return {
        "X_train": X_scaled[train_idx],
        "y_train": y_scaled[train_idx],
        "X_val": X_scaled[val_idx],
        "y_val": y_scaled[val_idx],
        "X_test": X_scaled[test_idx],
        "y_test": y_scaled[test_idx],
        "y_test_original": y[test_idx].ravel(),
        "test_meta": meta.iloc[test_idx].reset_index(drop=True),
        "scaler_x": scaler_x,
        "scaler_y": scaler_y,
        "feature_cols": feature_cols,
        "input_dim": len(feature_cols),
        "sequence_length": sequence_length,
    }
