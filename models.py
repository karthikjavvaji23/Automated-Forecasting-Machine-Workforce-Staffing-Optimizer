import numpy as np
import pandas as pd

from statsmodels.tsa.holtwinters import ExponentialSmoothing
from sklearn.linear_model import LinearRegression
from xgboost import XGBRegressor
from prophet import Prophet



def fit_predict_seasonal_naive(train_y, horizon, season):

    history = list(train_y)
    preds = []

    for i in range(horizon):

        if len(history) >= season:
            pred = history[-season]
        else:
            pred = history[-1]

        preds.append(pred)
        history.append(pred)

    return np.array(preds)


def fit_predict_ets(train_y, horizon, season):

    model = ExponentialSmoothing(
        train_y,
        trend=None,
        seasonal="add",
        seasonal_periods=season
    ).fit()

    forecast = model.forecast(horizon)

    return np.array(forecast)


def fit_predict_moving_avg(train_y, horizon, window):

    history = list(train_y)
    preds = []

    for i in range(horizon):

        pred = np.mean(history[-window:])
        preds.append(pred)

        history.append(pred)

    return np.array(preds)

def fit_predict_linear(train_feat, test_feat):

    X_train = train_feat.drop(columns=["ds", "y"])
    y_train = train_feat["y"]

    X_test = test_feat.drop(columns=["ds", "y"])

    model = LinearRegression()
    model.fit(X_train, y_train)

    preds = model.predict(X_test)

    return np.array(preds)


from xgboost import XGBRegressor
import numpy as np


def fit_predict_xgb(train_df, test_df, params=None):

  
    X_train = train_df.drop(columns=["ds", "y"])
    y_train = train_df["y"]

    X_test = test_df.drop(columns=["ds", "y"])

  
    model = XGBRegressor(
        n_estimators=300,
        max_depth=6,
        learning_rate=0.03,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42
    )


    model.fit(X_train, y_train)

 
    preds = model.predict(X_test)


    preds = np.maximum(0, preds)

    return preds


from prophet import Prophet
import pandas as pd
import numpy as np


def fit_predict_prophet(train_df, horizon):

    
    df = train_df.copy()

    df["ds"] = pd.to_datetime(df["ds"])
    df = df[["ds", "y"]]  

    model = Prophet(
        weekly_seasonality=True,
        daily_seasonality=False,
        yearly_seasonality=False
    )

    model.fit(df)

    future = model.make_future_dataframe(periods=horizon)

    forecast = model.predict(future)

    preds = forecast["yhat"].tail(horizon).values

    return np.array(preds)