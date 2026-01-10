import pandas as pd
import numpy as np
import os
from datetime import datetime
from typing import List, Dict, Optional

def load_price_data(
    csv_path: str,
    resolution: str = "15min",
    time_range: tuple[str, str] | None = None,
):
    """
    Load day-ahead electricity prices from Energy-Charts CSV and return:
        - df_resampled: the full (or sliced) resampled DataFrame
        - price_series: numpy array (float32)
        - timestamps: pandas DatetimeIndex

    Source:
        Energy-Charts (Fraunhofer ISE):
        https://www.energy-charts.info
        
    Parameters
    ----------
    csv_path : str
        Path to the CSV file.
    resolution : str
        Desired time resolution:
            - "15min" → high resolution
            - "1h"    → hourly data
    time_range : tuple(str, str) or None
        Optional time slicing, e.g. ("2025-11-01", "2025-11-07")
        If None → no slicing is applied and the full time span is used.

    Returns
    -------
    df_resampled : pd.DataFrame
        Resampled dataframe (possibly time-sliced)
    price_series : np.ndarray
        Float32 numpy array of prices
    timestamps : pd.DatetimeIndex
        Timestamp index matching the price series
    """

    # --- Load CSV and parse datetime ---
    df = pd.read_csv(
        csv_path,
        parse_dates=["Datum (MEZ)"],    # adjust column name if needed
    )

    # Set datetime index
    df = df.set_index("Datum (MEZ)").sort_index()

    # Dynamically detect the Day-Ahead price column
    price_col = None
    for col in df.columns:
        if "Day" in col and "Ahead" in col:
            price_col = col
            break

    if price_col is None:
        raise ValueError("Could not find 'Day Ahead' price column in CSV.")

    # --- Optional: time slicing BEFORE resampling ---
    if time_range is not None:
        start, end = time_range
        df = df.loc[start:end]

    # --- Resampling ---
    res = resolution.lower()
    if res in ["1h", "1hour", "hour"]:
        df_resampled = df.resample("1h").mean()
    elif res in ["15min", "15m", "quarter"]:
        df_resampled = df.resample("15min").mean()
    else:
        raise ValueError("resolution must be '1h' or '15min'")

    # Extract price series and drop NaNs
    prices = df_resampled[price_col].dropna().astype(np.float32)

    price_series = prices.values.astype(np.float32)
    timestamps = prices.index

    return df_resampled, price_series, timestamps


def load_demand_data(
    csv_path: str,
    resolution: str = "15min",
    time_range: tuple[str, str] | None = None,
    tz: str = "Europe/Berlin",
    demand_col_mode: str = "netzlast",  # "netzlast" | "netzlast_inkl_pumpspeicher" | "pumpspeicher" | "residuallast"
):
    """
    Load electricity demand (SMARD-like CSV) and return:
        - df_resampled: the full (or sliced) resampled DataFrame
        - demand_series: numpy array (float32) (average power in MW)
        - timestamps: pandas DatetimeIndex

    Source:
        SMARD – Bundesnetzagentur:
        https://www.smard.de
        
    Parameters
    ----------
    csv_path : str
        Path to the demand CSV.
    resolution : str
        Desired time resolution:
            - "15min" → 15-minute data
            - "1h"    → hourly data
    time_range : tuple(str, str) or None
        Optional slicing, e.g. ("2025-11-01", "2025-11-07") (inclusive)
    tz : str
        Timezone used for localization (Germany: Europe/Berlin).
    demand_col_mode : str
        Which demand-like column to extract:
            - "netzlast" (recommended as demand)
            - "netzlast_inkl_pumpspeicher"
            - "pumpspeicher"
            - "residuallast"

    Returns
    -------
    df_resampled : pd.DataFrame
        Resampled dataframe (possibly time-sliced)
    demand_series : np.ndarray
        Float32 numpy array of demand in MW (average over interval)
    timestamps : pd.DatetimeIndex
        Timestamp index matching the demand series
    """

    col_map = {
        "netzlast": "Netzlast [MWh] Originalauflösungen",
        "netzlast_inkl_pumpspeicher": "Netzlast inkl. Pumpspeicher [MWh] Originalauflösungen",
        "pumpspeicher": "Pumpspeicher [MWh] Originalauflösungen",
        "residuallast": "Residuallast [MWh] Originalauflösungen",
    }
    if demand_col_mode not in col_map:
        raise ValueError(f"Unknown demand_col_mode='{demand_col_mode}'. Choose one of {list(col_map.keys())}")

    df = pd.read_csv(
        csv_path,
        sep=";",
        decimal=",",
        thousands=".",
        encoding="utf-8",
    )

    required = {"Datum von", "Datum bis", col_map[demand_col_mode]}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")

    # Parse and localize timestamps
    t0 = pd.to_datetime(df["Datum von"], format="%d.%m.%Y %H:%M", errors="raise")
    t1 = pd.to_datetime(df["Datum bis"], format="%d.%m.%Y %H:%M", errors="raise")

    # Localize (DST-safe)
    t0 = t0.dt.tz_localize(tz, ambiguous="infer", nonexistent="shift_forward")
    t1 = t1.dt.tz_localize(tz, ambiguous="infer", nonexistent="shift_forward")

    dt_hours = (t1 - t0).dt.total_seconds() / 3600.0
    if (dt_hours <= 0).any():
        raise ValueError("Found non-positive intervals in 'Datum von'/'Datum bis'.")

    # Demand energy in interval (MWh), convert to average power (MW)
    demand_MWh = pd.to_numeric(df[col_map[demand_col_mode]], errors="coerce")
    if demand_MWh.isna().any():
        bad = int(demand_MWh.isna().sum())
        raise ValueError(f"Could not parse {bad} values in '{col_map[demand_col_mode]}'.")

    demand_MW = demand_MWh / dt_hours

    df["Datum (MEZ)"] = t0
    df["demand_MWh"] = demand_MWh
    df["demand_MW"] = demand_MW

    df = df.set_index("Datum (MEZ)").sort_index()

    # Optional slicing BEFORE resampling
    if time_range is not None:
        start, end = time_range
        df = df.loc[start:end]

    # Resampling
    res = resolution.lower()
    if res in ["1h", "1hour", "hour"]:
        df_resampled = df.resample("1h").mean(numeric_only=True)
    elif res in ["15min", "15m", "quarter"]:
        df_resampled = df.resample("15min").mean(numeric_only=True)
    else:
        raise ValueError("resolution must be '1h' or '15min'")

    demand = df_resampled["demand_MWh"].dropna().astype(np.float32)

    demand_series = demand.values.astype(np.float32)
    timestamps = demand.index

    return df_resampled, demand_series, timestamps



def save_records(
    records: List[Dict],
    out_path: str,
    agent_name: str,
    unique_cols: Optional[list] = None,
    add_timestamp: bool = True,
    experiment_id: Optional[str] = None,
):
    """
    Save experiment records to CSV with append + de-duplication.

    Parameters
    ----------
    records : list of dict
        List of experiment result records.
    out_path : str
        Path to CSV file (e.g. results/learning_steps_records.csv).
    agent_name : str
        Name of the agent (e.g. "DQN", "QRDQN", "TD3").
    unique_cols : list or None
        Columns defining uniqueness (default: ["agent", "training_steps", "run_id"]).
    add_timestamp : bool
        Whether to add a timestamp column.
    experiment_id : str or None
        Optional experiment identifier (same for all rows).
    """

    if unique_cols is None:
        unique_cols = ["agent", "training_steps", "run_id"]

    os.makedirs(os.path.dirname(out_path), exist_ok=True)

    df_new = pd.DataFrame(records)
    df_new["agent"] = agent_name

    if add_timestamp:
        df_new["timestamp"] = datetime.now().isoformat(timespec="seconds")

    if experiment_id is not None:
        df_new["experiment_id"] = experiment_id

    # Append if file exists
    if os.path.exists(out_path):
        df_old = pd.read_csv(out_path)
        df_all = pd.concat([df_old, df_new], ignore_index=True)
    else:
        df_all = df_new

    # Drop duplicates (keep latest)
    df_all = df_all.drop_duplicates(subset=unique_cols, keep="last")

    df_all.to_csv(out_path, index=False)

    print(
        f"[save_experiment_records] Saved {len(df_new)} new rows "
        f"(total={len(df_all)}) to {out_path}"
    )

    return df_all
