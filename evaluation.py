from __future__ import annotations

from typing import Dict, Optional

import numpy as np
import pandas as pd
from sklearn.metrics import explained_variance_score, mean_absolute_error, mean_squared_error, r2_score


def regression_metrics(y_true, y_pred, high_threshold: Optional[float] = None, low_threshold: Optional[float] = None) -> Dict[str, float]:
    """Compute standard and extreme-tail regression metrics."""
    y_true = np.asarray(y_true, dtype=float).ravel()
    y_pred = np.asarray(y_pred, dtype=float).ravel()
    mask = np.isfinite(y_true) & np.isfinite(y_pred)
    y_true = y_true[mask]
    y_pred = y_pred[mask]
    if len(y_true) == 0:
        raise ValueError("No finite y_true/y_pred pairs available for metric calculation.")

    rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
    mae = float(mean_absolute_error(y_true, y_pred))
    r2 = float(r2_score(y_true, y_pred)) if len(y_true) > 1 else np.nan
    ev = float(explained_variance_score(y_true, y_pred)) if len(y_true) > 1 else np.nan
    corr = float(np.corrcoef(y_true, y_pred)[0, 1]) if len(y_true) > 1 else np.nan
    bias = float(np.mean(y_pred - y_true))

    if high_threshold is None:
        high_threshold = float(np.percentile(y_true, 95))
    if low_threshold is None:
        low_threshold = float(np.percentile(y_true, 5))

    high_mask = y_true >= high_threshold
    low_mask = y_true <= low_threshold
    extreme_high_rmse = float(np.sqrt(mean_squared_error(y_true[high_mask], y_pred[high_mask]))) if np.any(high_mask) else np.nan
    extreme_low_rmse = float(np.sqrt(mean_squared_error(y_true[low_mask], y_pred[low_mask]))) if np.any(low_mask) else np.nan

    return {
        "n": int(len(y_true)),
        "rmse": rmse,
        "mae": mae,
        "r2": r2,
        "explained_variance": ev,
        "correlation": corr,
        "bias": bias,
        "extreme_high_rmse": extreme_high_rmse,
        "extreme_low_rmse": extreme_low_rmse,
        "high_threshold": float(high_threshold),
        "low_threshold": float(low_threshold),
        "n_high": int(high_mask.sum()),
        "n_low": int(low_mask.sum()),
    }


def metrics_table(results: Dict[str, Dict[str, float]]) -> pd.DataFrame:
    """Convert model results dictionary to a table."""
    rows = []
    for name, metrics in results.items():
        row = {"model": name}
        row.update(metrics)
        rows.append(row)
    return pd.DataFrame(rows)


def skill_against_persistence(metrics_df: pd.DataFrame, persistence_name: str = "Persistence") -> pd.DataFrame:
    """Add percentage RMSE skill relative to persistence."""
    df = metrics_df.copy()
    if persistence_name in set(df["model"]):
        pers_rmse = float(df.loc[df["model"] == persistence_name, "rmse"].iloc[0])
        df["rmse_skill_vs_persistence_%"] = 100.0 * (pers_rmse - df["rmse"]) / pers_rmse
    return df
