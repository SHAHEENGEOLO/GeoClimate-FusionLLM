from __future__ import annotations

from pathlib import Path
from typing import Optional

import pandas as pd

from .evaluation import regression_metrics


def validate_against_station(weather_df: pd.DataFrame, station_path: Optional[str | Path], output_dir: str | Path) -> Optional[pd.DataFrame]:
    """
    Compare blended NWP-derived tempmax with independent station observations.

    The station file must contain datetime and tempmax_obs. If no station file is
    supplied, a limitation note is written for the manuscript response.
    """
    output_dir = Path(output_dir)
    report_dir = output_dir / "reports"
    table_dir = output_dir / "tables"
    report_dir.mkdir(parents=True, exist_ok=True)
    table_dir.mkdir(parents=True, exist_ok=True)

    if station_path is None or not Path(station_path).exists():
        note = (
            "# Station-observation validation note\n\n"
            "No independent Baghdad station-observation file was provided. The manuscript should therefore state that the model is evaluated against a blended NWP-derived reference product, not against independent in-situ observations. Future validation should merge quality-controlled station tempmax observations by date and report bias, MAE, RMSE, correlation, and extreme-tail errors.\n"
        )
        (report_dir / "station_validation_note.md").write_text(note, encoding="utf-8")
        return None

    station = pd.read_csv(station_path)
    if "datetime" not in station.columns or "tempmax_obs" not in station.columns:
        raise ValueError("Station file must contain columns: datetime,tempmax_obs")
    station["datetime"] = pd.to_datetime(station["datetime"])
    merged = weather_df[["datetime", "tempmax"]].merge(station[["datetime", "tempmax_obs"]], on="datetime", how="inner")
    metrics = regression_metrics(merged["tempmax_obs"].values, merged["tempmax"].values)
    out = pd.DataFrame([{ "comparison": "Blended product vs station observation", **metrics }])
    out.to_csv(table_dir / "station_validation_metrics.csv", index=False)
    return out
