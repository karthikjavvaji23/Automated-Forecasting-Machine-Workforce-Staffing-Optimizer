import pandas as pd
import numpy as np
import plotly.express as px


def monthly_trend(ts):
    df = ts.copy()
    df["ds"] = pd.to_datetime(df["ds"])

    # month-year bins (MS = month start)
    monthly = (
        df.set_index("ds")["y"]
        .resample("MS")
        .sum()
        .reset_index()
        .rename(columns={"y": "monthly_volume"})
    )

    fig = px.bar(
        monthly,
        x="ds",
        y="monthly_volume",
        title="Monthly Demand (Month-Year)"
    )
    fig.update_xaxes(dtick="M1", tickformat="%b %Y")
    fig.update_layout(xaxis_title="Month", yaxis_title="Total Volume")

    return fig, monthly


def hourly_pattern(ts):
    df = ts.copy()
    df["ds"] = pd.to_datetime(df["ds"])
    df["hour"] = df["ds"].dt.hour

    hourly = df.groupby("hour")["y"].mean().reset_index(name="avg_hourly_volume")

    fig = px.line(
        hourly,
        x="hour",
        y="avg_hourly_volume",
        title="Average Demand by Hour of Day"
    )
    fig.update_layout(xaxis_title="Hour", yaxis_title="Avg Volume (per hour)")

    return fig, hourly


def dow_pattern(ts):
    df = ts.copy()
    df["ds"] = pd.to_datetime(df["ds"])
    df["dow"] = df["ds"].dt.day_name()

    order = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
    dow = (
        df.groupby("dow")["y"].mean()
        .reindex(order)
        .reset_index(name="avg_volume")
    )

    fig = px.bar(
        dow,
        x="dow",
        y="avg_volume",
        title="Average Demand by Day of Week"
    )
    fig.update_layout(xaxis_title="Day", yaxis_title="Avg Volume (per hour)")

    return fig, dow


def top_categories(df, cat_col, top_n=10):
    counts = df[cat_col].astype(str).value_counts().head(top_n).reset_index()
    counts.columns = [cat_col, "count"]

    fig = px.bar(
        counts,
        x="count",
        y=cat_col,
        orientation="h",
        title="Top Demand Categories (Row Count)"
    )
    fig.update_layout(xaxis_title="Rows", yaxis_title="Category")

    return fig, counts


def generate_insights(ts, monthly_df, hourly_df, dow_df, top_cat_df=None, cat_col=None):
    
    insights = []
    df = ts.copy()
    df["ds"] = pd.to_datetime(df["ds"])

    peak_month_row = monthly_df.loc[monthly_df["monthly_volume"].idxmax()]
    peak_month = peak_month_row["ds"].strftime("%b %Y")
    peak_month_vol = int(peak_month_row["monthly_volume"])
    insights.append(f"Peak month was **{peak_month}** with **{peak_month_vol:,}** total requests.")

 
    daily = df.set_index("ds")["y"].resample("D").sum()
    if len(daily) >= 56:
        last_28 = daily.iloc[-28:].mean()
        prev_28 = daily.iloc[-56:-28].mean()
        if prev_28 > 0:
            pct = (last_28 - prev_28) / prev_28 * 100
            direction = "increased" if pct >= 0 else "decreased"
            insights.append(f"Demand has **{direction} {abs(pct):.1f}%** (last 4 weeks vs previous 4 weeks).")

    
    peak_hour_row = hourly_df.loc[hourly_df["avg_hourly_volume"].idxmax()]
    peak_hour = int(peak_hour_row["hour"])
    peak_hour_val = float(peak_hour_row["avg_hourly_volume"])
    insights.append(f"Peak hour is around **{peak_hour:02d}:00** with **{peak_hour_val:.1f} avg requests/hour**.")

 
    peak_day_row = dow_df.loc[dow_df["avg_volume"].idxmax()]
    peak_day = str(peak_day_row["dow"])
    peak_day_val = float(peak_day_row["avg_volume"])
    insights.append(f"Highest day is **{peak_day}** with **{peak_day_val:.1f} avg requests/hour**.")

  
    avg = float(df["y"].mean())
    p95 = float(df["y"].quantile(0.95))
    insights.append(f"Variability: avg **{avg:.1f}/hr**, 95th percentile **{p95:.0f}/hr** (use this for buffer planning).")

  
    if top_cat_df is not None and cat_col and not top_cat_df.empty:
        top1 = top_cat_df.iloc[0]
        insights.append(f"Top driver: **{top1[cat_col]}** contributes **{int(top1['count']):,}** requests (top category).")


    insights.append("WFM note: schedule more coverage during peak hour + peak day windows; use P95 as a service-level buffer reference.")

    return insights