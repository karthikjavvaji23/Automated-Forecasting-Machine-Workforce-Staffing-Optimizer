import pandas as pd
import numpy as np

def make_lag_features(df, max_lag=168):

    df = df.copy()



    df["hour"] = df["ds"].dt.hour
    df["dayofweek"] = df["ds"].dt.dayofweek
    df["month"] = df["ds"].dt.month
    df["weekofyear"] = df["ds"].dt.isocalendar().week.astype(int)

    df["is_weekend"] = (df["dayofweek"] >= 5).astype(int)



    df["hour_sin"] = np.sin(2 * np.pi * df["hour"] / 24)
    df["hour_cos"] = np.cos(2 * np.pi * df["hour"] / 24)

    df["dow_sin"] = np.sin(2 * np.pi * df["dayofweek"] / 7)
    df["dow_cos"] = np.cos(2 * np.pi * df["dayofweek"] / 7)



    lags = [1, 2, 3, 24, 48, 72, 168]

    for lag in lags:
        df[f"lag_{lag}"] = df["y"].shift(lag)



    df["roll_24_mean"] = df["y"].shift(1).rolling(24).mean()
    df["roll_24_std"] = df["y"].shift(1).rolling(24).std()

    df["roll_168_mean"] = df["y"].shift(1).rolling(168).mean()


    df = df.dropna()

    return df