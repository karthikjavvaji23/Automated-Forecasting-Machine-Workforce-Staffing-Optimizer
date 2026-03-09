import pandas as pd
import numpy as np


def build_hourly_demand_continuous(df: pd.DataFrame, date_col: str) -> pd.DataFrame:
    d = df.copy()
    d[date_col] = pd.to_datetime(d[date_col], errors="coerce")
    d = d.dropna(subset=[date_col]).sort_values(date_col)

    d["_ds"] = d[date_col].dt.floor("H")

    hourly = d.groupby("_ds").size().rename("y").to_frame()
    hourly.index.name = "ds"

    full_idx = pd.date_range(hourly.index.min(), hourly.index.max(), freq="H")
    hourly = hourly.reindex(full_idx, fill_value=0)
    hourly.index.name = "ds"

    out = hourly.reset_index()
    out["freq"] = "H"
    return out


def clean_data(df: pd.DataFrame, date_col: str) -> pd.DataFrame:
    df = df.copy()
    df[date_col] = pd.to_datetime(df[date_col], errors="coerce")
    df = df.dropna(subset=[date_col])
    df = df.sort_values(date_col)
    return df


def build_hourly_demand(df: pd.DataFrame, date_col: str) -> pd.DataFrame:
    demand = (
        df.set_index(date_col)
        .resample("H")
        .size()
        .rename("y")
        .reset_index()
    )
    demand = demand.rename(columns={date_col: "ds"})
    return demand



def _mad(x: np.ndarray) -> float:
    med = np.nanmedian(x)
    return float(np.nanmedian(np.abs(x - med)))


def cap_outliers_mad(
    ts: pd.DataFrame,
    col: str = "y",
    z_thresh: float = 6.0,
) -> pd.DataFrame:
    """
    Robust capping using MAD-based z-score:
    z = 0.6745*(x - median)/MAD
    Caps values beyond +/- z_thresh to the threshold value.
    """
    df = ts.copy()
    x = df[col].astype(float).values
    med = np.nanmedian(x)
    mad = _mad(x)

    if mad == 0 or np.isnan(mad):
        return df

    z = 0.6745 * (x - med) / mad

    upper_mask = z > z_thresh
    lower_mask = z < -z_thresh

    
    upper_cap = med + (z_thresh * mad / 0.6745)
    lower_cap = med - (z_thresh * mad / 0.6745)

    x = np.where(upper_mask, upper_cap, x)
    x = np.where(lower_mask, lower_cap, x)

    df[col] = x
    return df


def hampel_filter(
    ts: pd.DataFrame,
    col: str = "y",
    window: int = 24,
    n_sigmas: float = 4.0,
) -> pd.DataFrame:
    
    df = ts.copy()
    s = df[col].astype(float)

    roll_med = s.rolling(window=window, center=True, min_periods=1).median()

   
    def rolling_mad(x):
        med = np.median(x)
        return np.median(np.abs(x - med))

    roll_mad = s.rolling(window=window, center=True, min_periods=1).apply(rolling_mad, raw=True)

  
    scale = 1.4826 * roll_mad.replace(0, np.nan)
    diff = (s - roll_med).abs()

    outlier = diff > (n_sigmas * scale)
    df.loc[outlier, col] = roll_med.loc[outlier]

    return df


def smooth_series(ts: pd.DataFrame, window: int = 3) -> pd.DataFrame:
    df = ts.copy()
    df["y"] = df["y"].rolling(window=window, center=True, min_periods=1).mean()
    return df


def clean_demand_series(
    ts: pd.DataFrame,
    use_mad_cap: bool = True,
    mad_z: float = 6.0,
    use_hampel: bool = True,
    hampel_window: int = 24,
    hampel_sigmas: float = 4.0,
    smooth_window: int = 3,
) -> pd.DataFrame:
   
    df = ts.copy()

    if use_mad_cap:
        df = cap_outliers_mad(df, col="y", z_thresh=mad_z)

    if use_hampel:
        df = hampel_filter(df, col="y", window=hampel_window, n_sigmas=hampel_sigmas)

    if smooth_window and smooth_window > 1:
        df = smooth_series(df, window=smooth_window)

  
    df["y"] = df["y"].clip(lower=0)

    return df