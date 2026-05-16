from __future__ import annotations

from pathlib import Path
from typing import Dict, Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats
from statsmodels.graphics.tsaplots import plot_acf

from .utils import ensure_dir


def save_framework_diagram(path: str | Path) -> None:
    """Create a reproducible conceptual framework diagram."""
    path = Path(path)
    fig, ax = plt.subplots(figsize=(14, 8), dpi=300)
    ax.axis("off")
    boxes = [
        (0.05, 0.70, 0.20, 0.16, "Multi-source\nmeteorological records"),
        (0.30, 0.70, 0.20, 0.16, "Leakage-controlled\nfeature engineering"),
        (0.55, 0.70, 0.20, 0.16, "Hybrid forecasting\nMMWSTM-ADRAN+"),
        (0.80, 0.70, 0.15, 0.16, "Warnings\n& decisions"),
        (0.30, 0.35, 0.20, 0.16, "Simple baselines\nPersistence / AR / MLR"),
        (0.55, 0.35, 0.20, 0.16, "Ablations\n+ seed robustness"),
        (0.80, 0.35, 0.15, 0.16, "LLM/RAG-style\nexplanation report"),
    ]
    for x, y, w, h, text in boxes:
        ax.add_patch(plt.Rectangle((x, y), w, h, fill=False, linewidth=2))
        ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=12, weight="bold")
    arrows = [((0.25, 0.78), (0.30, 0.78)), ((0.50, 0.78), (0.55, 0.78)), ((0.75, 0.78), (0.80, 0.78)), ((0.40, 0.70), (0.40, 0.51)), ((0.65, 0.70), (0.65, 0.51)), ((0.75, 0.43), (0.80, 0.43))]
    for (x1, y1), (x2, y2) in arrows:
        ax.annotate("", xy=(x2, y2), xytext=(x1, y1), arrowprops=dict(arrowstyle="->", lw=2))
    ax.set_title("Figure 1. GeoClimate-FusionLLM reproducible early-warning framework", fontsize=16, weight="bold")
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


def save_feature_cube(df: pd.DataFrame, feature_cols, path: str | Path) -> None:
    """Create a feature-correlation heatmap as a multimodal feature cube proxy."""
    path = Path(path)
    cols = [c for c in feature_cols[:25] if c in df.columns]
    corr = df[cols].corr().fillna(0)
    fig, ax = plt.subplots(figsize=(12, 10), dpi=300)
    im = ax.imshow(corr.values, aspect="auto")
    ax.set_xticks(range(len(cols)))
    ax.set_yticks(range(len(cols)))
    ax.set_xticklabels(cols, rotation=90, fontsize=7)
    ax.set_yticklabels(cols, fontsize=7)
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label="Correlation")
    ax.set_title("Figure 2. Multi-modal engineered meteorological feature cube", fontsize=14, weight="bold")
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def save_3d_temperature_surface(df: pd.DataFrame, path: str | Path) -> None:
    """Create a 3D year-month-temperature surface/scatter figure."""
    path = Path(path)
    work = df.copy()
    work["year_decimal"] = pd.to_datetime(work["datetime"]).dt.year + (pd.to_datetime(work["datetime"]).dt.dayofyear - 1) / 365.25
    fig = plt.figure(figsize=(12, 9), dpi=300)
    ax = fig.add_subplot(111, projection="3d")
    sc = ax.scatter(work["year_decimal"], work["month"], work["tempmax"], c=work["tempmax"], s=10, alpha=0.80)
    ax.set_xlabel("Year")
    ax.set_ylabel("Month")
    ax.set_zlabel("Daily maximum temperature (°C)")
    ax.set_title("Figure 3. 3D temporal distribution of Baghdad maximum temperature", weight="bold")
    fig.colorbar(sc, ax=ax, shrink=0.65, label="Tempmax (°C)")
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


def save_model_performance(metrics: pd.DataFrame, path: str | Path) -> None:
    """Save bar chart for overall model performance."""
    path = Path(path)
    df = metrics.copy().sort_values("rmse")
    fig, ax = plt.subplots(figsize=(12, 6), dpi=300)
    ax.bar(df["model"], df["rmse"])
    ax.set_ylabel("RMSE (°C)")
    ax.set_title("Figure 4. Overall test-period RMSE comparison", weight="bold")
    ax.tick_params(axis="x", rotation=35)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def save_extreme_tail_performance(metrics: pd.DataFrame, path: str | Path) -> None:
    """Save hot/cold extreme-tail RMSE comparison."""
    path = Path(path)
    df = metrics.copy()
    x = np.arange(len(df))
    width = 0.38
    fig, ax = plt.subplots(figsize=(13, 6), dpi=300)
    ax.bar(x - width / 2, df["extreme_high_rmse"], width, label="Hot tail RMSE")
    ax.bar(x + width / 2, df["extreme_low_rmse"], width, label="Cold tail RMSE")
    ax.set_xticks(x)
    ax.set_xticklabels(df["model"], rotation=35, ha="right")
    ax.set_ylabel("RMSE (°C)")
    ax.set_title("Figure 5. Extreme-tail forecast performance", weight="bold")
    ax.legend()
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def save_predictions_vs_actual(test_meta: pd.DataFrame, y_true, predictions: Dict[str, np.ndarray], path: str | Path, max_models: int = 4) -> None:
    """Save time-series plot of actual vs selected model predictions."""
    path = Path(path)
    dates = pd.to_datetime(test_meta["target_date"])
    fig, ax = plt.subplots(figsize=(14, 6), dpi=300)
    ax.plot(dates, y_true, label="Reference tempmax", linewidth=2)
    for i, (name, pred) in enumerate(predictions.items()):
        if i >= max_models:
            break
        ax.plot(dates, pred, label=name, alpha=0.85, linewidth=1)
    ax.set_ylabel("Tempmax (°C)")
    ax.set_title("Figure 6. One-day-ahead predictions versus reference product", weight="bold")
    ax.legend(ncol=2)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def save_residual_diagnostics(y_true, y_pred, path: str | Path) -> None:
    """Create residual diagnostics figure."""
    path = Path(path)
    resid = np.asarray(y_pred) - np.asarray(y_true)
    fig = plt.figure(figsize=(12, 10), dpi=300)
    gs = fig.add_gridspec(2, 2)
    ax1 = fig.add_subplot(gs[0, 0])
    ax1.scatter(y_pred, resid, s=10, alpha=0.7)
    ax1.axhline(0, linestyle="--", linewidth=1)
    ax1.set_xlabel("Predicted (°C)")
    ax1.set_ylabel("Residual (°C)")
    ax1.set_title("Residuals vs predicted")
    ax2 = fig.add_subplot(gs[0, 1])
    ax2.hist(resid, bins=25, edgecolor="black")
    ax2.set_title("Residual distribution")
    ax3 = fig.add_subplot(gs[1, 0])
    try:
        plot_acf(resid, ax=ax3, lags=30)
    except Exception:
        ax3.text(0.5, 0.5, "ACF unavailable", ha="center")
    ax4 = fig.add_subplot(gs[1, 1])
    stats.probplot(resid, dist="norm", plot=ax4)
    ax4.set_title("Q-Q plot")
    fig.suptitle("Figure 7. Residual diagnostics", fontsize=16, weight="bold")
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def save_ablation_robustness(ablation_metrics: pd.DataFrame, path: str | Path) -> None:
    """Save ablation robustness bar chart from seed-level results."""
    path = Path(path)
    if ablation_metrics.empty:
        return
    summary = ablation_metrics.groupby("variant")["rmse"].agg(["mean", "std"]).reset_index().sort_values("mean")
    fig, ax = plt.subplots(figsize=(12, 6), dpi=300)
    ax.bar(summary["variant"], summary["mean"], yerr=summary["std"].fillna(0), capsize=4)
    ax.set_ylabel("RMSE mean ± SD (°C)")
    ax.set_title("Figure 8. Ablation and random-seed robustness", weight="bold")
    ax.tick_params(axis="x", rotation=30)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def create_all_static_figures(output_dir, df, feature_cols, metrics, test_meta, y_true, predictions, best_pred, ablation_metrics=None):
    """Create the full figure set used in the paper."""
    fig_dir = ensure_dir(Path(output_dir) / "figures")
    save_framework_diagram(fig_dir / "figure_01_framework.png")
    save_feature_cube(df, feature_cols, fig_dir / "figure_02_feature_cube.png")
    save_3d_temperature_surface(df, fig_dir / "figure_03_3d_temperature_surface.png")
    save_model_performance(metrics, fig_dir / "figure_04_model_performance.png")
    save_extreme_tail_performance(metrics, fig_dir / "figure_05_extreme_tail_performance.png")
    save_predictions_vs_actual(test_meta, y_true, predictions, fig_dir / "figure_06_predictions_vs_actual.png")
    save_residual_diagnostics(y_true, best_pred, fig_dir / "figure_07_residual_diagnostics.png")
    if ablation_metrics is not None and not ablation_metrics.empty:
        save_ablation_robustness(ablation_metrics, fig_dir / "figure_08_ablation_robustness.png")
