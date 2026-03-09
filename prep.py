import pandas as pd

def infer_frequency(index: pd.DatetimeIndex) -> str:
    freq = pd.infer_freq(index[:500]) if len(index) >= 10 else None
    if freq:
        return freq

    diffs = index.to_series().diff().dropna()
    if diffs.empty:
        return "D"
    med = diffs.median()

    if med <= pd.Timedelta(hours=2):
        return "H"
    if med <= pd.Timedelta(days=1.5):
        return "D"
    if med <= pd.Timedelta(days=10):
        return "W"
    return "MS"

def to_timeseries(df: pd.DataFrame, dt_col: str, target_col: str | None) -> pd.DataFrame:
    d = df.copy()
    d[dt_col] = pd.to_datetime(d[dt_col], errors="coerce")
    d = d.dropna(subset=[dt_col]).sort_values(dt_col)

    if target_col is None:
        d["__y__"] = 1
        target_col = "__y__"
    else:
        d[target_col] = pd.to_numeric(d[target_col], errors="coerce")
        d = d.dropna(subset=[target_col])

    d = d[[dt_col, target_col]].rename(columns={dt_col: "ds", target_col: "y"})
    d = d.set_index("ds")
    d = d[~d.index.duplicated(keep="last")]

    freq = infer_frequency(d.index)

    y = d["y"].resample(freq).sum().fillna(0.0)
    out = y.to_frame("y")
    out.index.name = "ds"
    out.reset_index(inplace=True)
    out["freq"] = freq
    return out