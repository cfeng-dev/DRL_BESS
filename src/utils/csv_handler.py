import pandas as pd
import numpy as np


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
