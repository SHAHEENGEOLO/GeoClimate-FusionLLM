from __future__ import annotations

from pathlib import Path
from typing import Optional

import pandas as pd


def load_weather_data(raw_path: str | Path, processed_path: Optional[str | Path] = None, prefer_processed: bool = False) -> pd.DataFrame:
    """
    Load Baghdad weather data from Excel or CSV.

    Parameters
    ----------
    raw_path:
        Path to the raw Excel/CSV file.
    processed_path:
        Optional path to a previously processed CSV.
    prefer_processed:
        If True and processed_path exists, load the processed file. The default
        is False to keep preprocessing reproducible from raw data.
    """
    if prefer_processed and processed_path is not None and Path(processed_path).exists():
        df = pd.read_csv(processed_path)
    else:
        raw_path = Path(raw_path)
        if not raw_path.exists():
            raise FileNotFoundError(f"Input file not found: {raw_path}")
        if raw_path.suffix.lower() in {".xlsx", ".xls"}:
            df = pd.read_excel(raw_path)
        elif raw_path.suffix.lower() == ".csv":
            df = pd.read_csv(raw_path)
        else:
            raise ValueError(f"Unsupported file extension: {raw_path.suffix}")

    df = clean_weather_frame(df)
    return df


def clean_weather_frame(df: pd.DataFrame) -> pd.DataFrame:
    """Clean column names, remove empty columns, parse datetime, and sort."""
    df = df.copy()
    df.columns = [str(c).strip() for c in df.columns]

    # Remove unnamed or fully empty columns.
    drop_cols = [c for c in df.columns if c.lower().startswith("unnamed")]
    drop_cols += [c for c in df.columns if df[c].isna().all()]
    df = df.drop(columns=list(set(drop_cols)), errors="ignore")

    if "datetime" not in df.columns:
        raise ValueError("The weather dataset must contain a 'datetime' column.")
    df["datetime"] = pd.to_datetime(df["datetime"], errors="coerce")
    df = df.dropna(subset=["datetime"]).sort_values("datetime").reset_index(drop=True)

    if "tempmax" not in df.columns:
        raise ValueError("The weather dataset must contain the target column 'tempmax'.")

    # Convert possible numeric columns stored as strings.
    for col in df.columns:
        if col == "datetime":
            continue
        if df[col].dtype == object:
            try:
                converted = pd.to_numeric(df[col])
                if converted.notna().sum() > 0:
                    df[col] = converted
            except Exception:
                pass

    return df
