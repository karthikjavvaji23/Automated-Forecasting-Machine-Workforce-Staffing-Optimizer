import pandas as pd

COMMON_DATE_HINTS = ["date", "datetime", "timestamp", "created", "time"]

def guess_datetime_column(df: pd.DataFrame) -> str | None:
    for c in df.columns:
        cl = str(c).lower()
        if any(h in cl for h in COMMON_DATE_HINTS):
            return c

    for c in df.columns:
        if df[c].dtype == "object":
            sample = df[c].dropna().astype(str).head(200)
            if sample.empty:
                continue
            parsed = pd.to_datetime(sample, errors="coerce", utc=False, infer_datetime_format=True)
            if parsed.notna().mean() > 0.80:
                return c
    return None

def guess_target_column(df: pd.DataFrame, dt_col: str | None) -> str | None:
    candidates = []
    for c in df.columns:
        if c == dt_col:
            continue
        if pd.api.types.is_numeric_dtype(df[c]):
            candidates.append(c)

    if candidates:
        candidates.sort(key=lambda c: df[c].notna().sum(), reverse=True)
        return candidates[0]
    return None