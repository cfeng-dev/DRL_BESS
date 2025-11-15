import pandas as pd
import numpy as np


def load_price_data(csv_path: str, resolution: str = "1H"):
    """
    Load day-ahead electricity prices from Energy-Charts CSV and return:
        - price_series: numpy array (float32)
        - timestamps: pandas DateTimeIndex

    Parameters
    ----------
    csv_path : str
        Path to the CSV file downloaded from Energy-Charts.
    resolution : str
        Desired time resolution. Choose:
            - "15min"  → high-resolution market data
            - "1H"     → hourly data (recommended for RL env with dt_hours=1.0)

    Returns
    -------
    price_series : np.ndarray
        Normalized or raw price data as float32 array.
    timestamps : pd.DatetimeIndex
        Timestamps aligned with the price_series.
    """

    # --- Load CSV and parse datetime ---
    df = pd.read_csv(
        csv_path,
        parse_dates=["Datum (MEZ)"]
    )

    # Set datetime as index
    df = df.set_index("Datum (MEZ)")

    # Detect price column name dynamically
    price_col = None
    for col in df.columns:
        if "Day" in col and "Ahead" in col:
            price_col = col
            break

    if price_col is None:
        raise ValueError("Could not find 'Day Ahead' price column in CSV.")

    # --- Resample to desired resolution ---
    if resolution.lower() in ["1h", "1hour", "hour"]:
        df_resampled = df.resample("1h").mean()
    elif resolution.lower() in ["15min", "15m", "quarter"]:
        df_resampled = df.resample("15min").mean()
    else:
        raise ValueError("resolution must be '1h' or '15min'")

    # Extract price series
    prices = df_resampled[price_col].astype(np.float32)

    # Drop NaN rows (can appear after resampling)
    prices = prices.dropna()

    # Convert to numpy
    price_series = prices.values.astype(np.float32)
    timestamps = prices.index

    return df_resampled, price_series, timestamps
