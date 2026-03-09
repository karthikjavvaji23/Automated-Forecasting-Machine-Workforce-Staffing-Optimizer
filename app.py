import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px

from ingest import guess_datetime_column, guess_target_column
from prep import to_timeseries
from features import make_lag_features

from models import (
    fit_predict_seasonal_naive,
    fit_predict_ets,
    fit_predict_moving_avg,
    fit_predict_linear,
    fit_predict_xgb,
    fit_predict_prophet
)

from evaluate import wmape, mape, rmse, mae
from staffing import staffing_table

from eda import (
    monthly_trend,
    hourly_pattern,
    dow_pattern,
    top_categories,
    generate_insights
)


st.set_page_config(page_title="Forecasting Machine", layout="wide")

st.title("Forecasting Machine + Workforce Staffing Optimizer")


uploaded = st.file_uploader("Upload CSV dataset", type=["csv"])

if uploaded is None:
    st.info("Upload a dataset to begin")
    st.stop()

raw = pd.read_csv(uploaded, low_memory=False)

st.subheader("Raw Data Preview")
st.dataframe(raw.head())


dt_guess = guess_datetime_column(raw)
target_guess = guess_target_column(raw, dt_guess)

cat_guess = None
for c in raw.columns:
    if "problem" in c.lower() or "type" in c.lower() or "complaint" in c.lower():
        cat_guess = c
        break

c1, c2, c3, c4 = st.columns(4)

with c1:
    dt_col = st.selectbox(
        "Datetime column",
        raw.columns,
        index=list(raw.columns).index(dt_guess)
    )

with c2:
    target_opts = ["(none — count rows as demand)"] + list(raw.columns)
    target_sel = st.selectbox("Target column", target_opts)

with c3:
    horizon_choice = st.selectbox(
        "Forecast Horizon",
        ["3 months", "6 months"],
        index=1
    )

with c4:
    cat_opts = ["(none)"] + list(raw.columns)
    cat_sel = st.selectbox(
        "Category column",
        cat_opts,
        index=(cat_opts.index(cat_guess) if cat_guess in cat_opts else 0)
    )

target_col = None if target_sel.startswith("(none") else target_sel
cat_col = None if cat_sel == "(none)" else cat_sel


st.subheader("Data Inspection")

temp_df = raw.copy()
temp_df[dt_col] = pd.to_datetime(temp_df[dt_col], errors="coerce")

rows_count = temp_df.shape[0]
date_min = temp_df[dt_col].min()
date_max = temp_df[dt_col].max()

temp_df["hour"] = temp_df[dt_col].dt.hour

hour_unique = temp_df["hour"].nunique()
hour_min = temp_df["hour"].min()
hour_max = temp_df["hour"].max()

missing_dates = temp_df[dt_col].isna().sum()

inspection_table = pd.DataFrame({
    "Metric": [
        "Rows",
        "Date min",
        "Date max",
        "Hour unique count",
        "Hour min",
        "Hour max",
        "Missing Created Date"
    ],
    "Value": [
        rows_count,
        date_min,
        date_max,
        hour_unique,
        hour_min,
        hour_max,
        missing_dates
    ]
})

st.dataframe(inspection_table)


st.subheader("Dataset Structure")

dtype_df = pd.DataFrame({
    "Column": raw.columns,
    "Data Type": raw.dtypes.astype(str)
})

st.dataframe(dtype_df)


ts = to_timeseries(raw, dt_col, target_col)

freq = ts["freq"].iloc[0]
ts = ts.drop(columns=["freq"])


st.subheader("Data Cleaning")

clean_log = {}

missing_before = ts["y"].isna().sum()
ts = ts.dropna(subset=["y"])
missing_after = ts["y"].isna().sum()

clean_log["Missing values removed"] = missing_before - missing_after

median = ts["y"].median()
mad = np.median(np.abs(ts["y"] - median))

if mad == 0:
    lo, hi = ts["y"].quantile(0.01), ts["y"].quantile(0.99)
else:
    lo = median - 6 * mad
    hi = median + 6 * mad

before = ts["y"].copy()

ts["y"] = ts["y"].clip(lower=max(0, lo), upper=hi)

clean_log["Outliers capped"] = (before != ts["y"]).sum()

ts["y"] = ts["y"].rolling(3, center=True, min_periods=1).median()

clean_log["Smoothing applied"] = True

clean_df = pd.DataFrame(list(clean_log.items()), columns=["Step", "Result"])

st.dataframe(clean_df)


st.subheader("Historical Demand")

st.plotly_chart(
    px.line(ts, x="ds", y="y", title="Historical Demand"),
    use_container_width=True
)


with st.expander("Exploratory Data Analysis", expanded=True):

    fig_m, mdf = monthly_trend(ts)
    fig_h, hdf = hourly_pattern(ts)
    fig_d, ddf = dow_pattern(ts)

    c1, c2 = st.columns(2)

    with c1:
        st.plotly_chart(fig_m, use_container_width=True)
        st.plotly_chart(fig_d, use_container_width=True)

    with c2:
        st.plotly_chart(fig_h, use_container_width=True)

    top_df = None

    if cat_col:
        fig_c, top_df = top_categories(raw, cat_col)
        st.plotly_chart(fig_c, use_container_width=True)

    st.subheader("Automated Insights")

    insights = generate_insights(ts, mdf, hdf, ddf, top_df, cat_col)

    for i in insights:
        st.write("-", i)


months = 6 if horizon_choice == "6 months" else 3

if freq == "H":
    horizon = 24 * 30 * months
    season = 168
elif freq == "D":
    horizon = 30 * months
    season = 7
else:
    horizon = 30 * months
    season = 12

test_size = min(len(ts) // 6, 24 * 14 if freq == "H" else 28)

train = ts.iloc[:-test_size]
test = ts.iloc[-test_size:]

train_y = train["y"]
test_y = test["y"]


run = st.button("Run Model Comparison")

if not run:
    st.stop()

results = []
preds = {}

p = fit_predict_seasonal_naive(train_y, test_size, season)
results.append(("SeasonalNaive", wmape(test_y,p), mape(test_y,p), rmse(test_y,p), mae(test_y,p)))
preds["SeasonalNaive"] = p

try:
    p = fit_predict_ets(train_y, test_size, season)
    results.append(("ETS", wmape(test_y,p), mape(test_y,p), rmse(test_y,p), mae(test_y,p)))
    preds["ETS"] = p
except:
    pass

try:
    p = fit_predict_moving_avg(train_y, test_size, season)
    results.append(("MovingAverage", wmape(test_y,p), mape(test_y,p), rmse(test_y,p), mae(test_y,p)))
    preds["MovingAverage"] = p
except:
    pass

feat = make_lag_features(ts, 168)

cutoff = test["ds"].iloc[0]

train_feat = feat[feat["ds"] < cutoff]
test_feat = feat[feat["ds"] >= cutoff]

try:
    p = fit_predict_linear(train_feat, test_feat)
    results.append(("LinearRegression", wmape(test_feat["y"],p), mape(test_feat["y"],p), rmse(test_feat["y"],p), mae(test_feat["y"],p)))
    preds["LinearRegression"] = p
except:
    pass

try:
    params = dict(
        n_estimators=300,
        max_depth=6,
        learning_rate=0.03,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42
    )
    p = fit_predict_xgb(train_feat, test_feat, params)
    results.append(("XGBoost", wmape(test_feat["y"],p), mape(test_feat["y"],p), rmse(test_feat["y"],p), mae(test_feat["y"],p)))
    preds["XGBoost"] = p
except:
    pass

try:
    train_df = train[["ds","y"]]
    p = fit_predict_prophet(train_df, test_size)
    results.append(("Prophet", wmape(test_y,p), mape(test_y,p), rmse(test_y,p), mae(test_y,p)))
    preds["Prophet"] = p
except Exception as e:
    st.error(f"Prophet failed: {e}")

leaderboard = pd.DataFrame(results, columns=["Model","wMAPE","MAPE","RMSE","MAE"]).sort_values("wMAPE")

st.subheader("Model Leaderboard")
st.dataframe(leaderboard)

best_model = leaderboard.iloc[0]["Model"]

st.success(f"Best model selected: {best_model}")


st.subheader("Demand Forecast")

best_model_name = best_model

full_series = ts["y"]

if best_model_name == "SeasonalNaive":
    forecast = fit_predict_seasonal_naive(full_series, horizon, season)

elif best_model_name == "ETS":
    forecast = fit_predict_ets(full_series, horizon, season)

elif best_model_name == "MovingAverage":
    forecast = fit_predict_moving_avg(full_series, horizon, season)

elif best_model_name == "LinearRegression":

    feat_full = make_lag_features(ts, season)

    train_full = feat_full.iloc[:-horizon]
    future_feat = feat_full.iloc[-horizon:]

    forecast = fit_predict_linear(train_full, future_feat)

elif best_model_name == "XGBoost":

    feat_full = make_lag_features(ts, season)

    train_full = feat_full.iloc[:-horizon]
    future_feat = feat_full.iloc[-horizon:]

    params = dict(
        n_estimators=300,
        max_depth=6,
        learning_rate=0.03,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42
    )

    forecast = fit_predict_xgb(train_full, future_feat, params)

elif best_model_name == "Prophet":

    prophet_df = ts[["ds", "y"]]
    forecast = fit_predict_prophet(prophet_df, horizon)

forecast_index = pd.date_range(
    start=ts["ds"].iloc[-1],
    periods=horizon + 1,
    freq=freq
)[1:]

forecast_df = pd.DataFrame({
    "ds": forecast_index,
    "forecast": forecast
})


st.subheader("Forecast Visualization")

hist_df = ts.copy()

fig = px.line(
    hist_df,
    x="ds",
    y="y",
    title="Historical Demand"
)

fig.add_scatter(
    x=forecast_df["ds"],
    y=forecast_df["forecast"],
    mode="lines",
    name="Forecast",
)

st.plotly_chart(fig, use_container_width=True)


st.subheader("Forecast Summary Metrics")

avg_forecast = forecast_df["forecast"].mean()
peak_forecast = forecast_df["forecast"].max()
min_forecast = forecast_df["forecast"].min()

m1, m2, m3 = st.columns(3)

m1.metric("Average Demand", f"{avg_forecast:.1f}")
m2.metric("Peak Demand", f"{peak_forecast:.1f}")
m3.metric("Minimum Demand", f"{min_forecast:.1f}")


st.subheader("Workforce Staffing Optimizer")

aht = st.number_input("Average Handle Time (seconds)", value=360)

shrink = st.slider("Shrinkage", 0.0, 0.6, 0.30)

occ = st.slider("Occupancy", 0.5, 0.95, 0.85)

forecast = pd.DataFrame({
    "ds": forecast_df["ds"],
    "yhat": forecast_df["forecast"]
})

staff_df = staffing_table(
    forecast,
    interval_minutes=60,
    method="Workload",
    aht_seconds=aht,
    occupancy_cap=occ,
    shrinkage=shrink,
    target_sl=0.8,
    target_answer_seconds=20
)

st.dataframe(staff_df.head(200))

st.plotly_chart(
    px.line(
        staff_df,
        x="ds",
        y="required_agents",
        title="Required Staffing Over Time"
    ),
    use_container_width=True
)