import os
import numpy as np
import pandas as pd
import plotly.graph_objs as go
import streamlit as st

from models.arima_model import AutoARIMA
from models.prophet_model import ProphetModel

# ---------------------------------------------------
# STREAMLIT CONFIG
# ---------------------------------------------------
st.set_page_config(page_title="PM2.5 Forecasting Dashboard", layout="wide")

# ---------------------------------------------------
# DATA HELPERS
# ---------------------------------------------------
@st.cache_data(show_spinner=False)
def load_clean_series():
    df = pd.read_csv(os.path.join("data", "clean_pm25.csv"), parse_dates=["datetime"], index_col="datetime")
    if isinstance(df, pd.DataFrame) and df.shape[1] == 1:
        s = df.iloc[:, 0]
    else:
        s = df.squeeze()
    s.name = "pm25"
    return s.asfreq("D")

@st.cache_data(show_spinner=False)
def load_daily_weather():
    raw_path = os.path.join("data", "BeijingPM20100101_20151231.csv")
    if not os.path.exists(raw_path):
        return None

    raw = pd.read_csv(raw_path, na_values=["NA"])
    raw["datetime"] = pd.to_datetime(
        dict(year=raw["year"], month=raw["month"], day=raw["day"], hour=raw["hour"]), errors="coerce"
    )
    raw = raw.set_index("datetime").sort_index()
    candidates = [c for c in ["TEMP", "DEWP", "PRES", "WSPM", "Iws"] if c in raw.columns]
    if not candidates:
        return None
    daily_weather = raw.resample("D")[candidates].mean().ffill().bfill()
    if "Iws" in daily_weather.columns and "WSPM" not in daily_weather.columns:
        daily_weather = daily_weather.rename(columns={"Iws": "WSPM"})
    keep = [c for c in ["TEMP", "DEWP", "PRES", "WSPM"] if c in daily_weather.columns]
    return daily_weather[keep]

# ---------------------------------------------------
# VISUALIZATION HELPERS
# ---------------------------------------------------
def plot_series(history: pd.Series, test: pd.Series | None = None, preds: pd.Series | None = None, title=""):
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=history.index, y=history.values, name="History", mode="lines"))
    if test is not None:
        fig.add_trace(go.Scatter(x=test.index, y=test.values, name="Test (actual)", mode="lines"))
    if preds is not None:
        fig.add_trace(go.Scatter(x=preds.index, y=preds.values, name="Prediction", mode="lines"))
    fig.update_layout(height=420, title=title, xaxis_title="Date", yaxis_title="PM2.5 (µg/m³)")
    return fig

def plot_future(history: pd.Series, future_mean: pd.Series, low=None, high=None, title=""):
    fig = go.Figure()
    cutoff = history.index[-180] if len(history) >= 180 else history.index[0]
    fig.add_trace(go.Scatter(x=history.index[history.index >= cutoff],
                             y=history[history.index >= cutoff],
                             name="History (last 6m)",
                             mode="lines"))
    fig.add_trace(go.Scatter(x=future_mean.index, y=future_mean.values,
                             name="Forecast", mode="lines"))
    if low is not None and high is not None:
        fig.add_trace(go.Scatter(
            x=list(future_mean.index) + list(future_mean.index[::-1]),
            y=list(high.values) + list(low.values[::-1]),
            fill="toself",
            fillcolor="rgba(0,123,255,0.2)",
            line=dict(color="rgba(255,255,255,0)"),
            name="Confidence Interval",
            showlegend=True,
        ))
    fig.update_layout(height=420, title=title, xaxis_title="Date", yaxis_title="PM2.5 (µg/m³)")
    return fig

# ---------------------------------------------------
# SIDEBAR CONTROLS
# ---------------------------------------------------
st.sidebar.header("Controls")

horizon = st.sidebar.selectbox("Forecast horizon (days)", [7, 30, 60], index=1)
model_name = st.sidebar.selectbox(
    "Model",
    ["ARIMA (auto, non-seasonal)", "SARIMA (weekly m=7)", "Prophet (baseline)", "Prophet + Weather"],
    index=3
)
use_log = st.sidebar.checkbox("ARIMA: log1p transform", value=False)
cps = st.sidebar.slider("Prophet: changepoint_prior_scale", 0.05, 1.0, 0.2, 0.05)
seasonal_mode = st.sidebar.selectbox("Prophet: seasonality mode", ["additive", "multiplicative"], index=0)

st.sidebar.markdown("---")
st.sidebar.caption("The last N days of the series (same as horizon) are held out for evaluation.")

# ---------------------------------------------------
# LOAD DATA
# ---------------------------------------------------
daily = load_clean_series()
weather_df = load_daily_weather()

H = horizon
train = daily.iloc[:-H]
test = daily.iloc[-H:]

# ---------------------------------------------------
# HISTORICAL PLOT
# ---------------------------------------------------
st.title("🌫️ PM2.5 Forecasting Dashboard")
st.plotly_chart(plot_series(daily, title="Daily PM2.5 (µg/m³)"), use_container_width=True)

# ---------------------------------------------------
# MODEL FITTING AND FORECASTING
# ---------------------------------------------------
st.subheader("Model Fit & Evaluation")

try:
    # 1️⃣ Choose and instantiate model
    if model_name == "ARIMA (auto, non-seasonal)":
        model = AutoARIMA(seasonal=False, m=1, use_log=use_log)
    elif model_name == "SARIMA (weekly m=7)":
        model = AutoARIMA(seasonal=True, m=7, use_log=use_log)
    elif model_name == "Prophet (baseline)":
        model = ProphetModel(cps=cps, seasonal_mode=seasonal_mode)
    else:
        model = ProphetModel(cps=cps, seasonal_mode=seasonal_mode, regressors=weather_df)

    # 2️⃣ Evaluate on test and forecast future
    res = model.evaluate_and_forecast(daily, H)

    # 3️⃣ Display metrics
    c1, c2, c3 = st.columns(3)
    if res.metrics:
        c1.metric("MAE", f"{res.metrics['MAE']:.1f}")
        c2.metric("RMSE", f"{res.metrics['RMSE']:.1f}")
        if res.metrics["MAPE"] == res.metrics["MAPE"]:
            c3.metric("MAPE", f"{res.metrics['MAPE']:.1f}%")
        else:
            c3.metric("MAPE", "—")

    st.caption(f"Model: {res.info}")

    # 4️⃣ Plot test window overlay
    st.plotly_chart(
        plot_series(train, test, res.pred_test, title="Test Window — Actual vs Prediction"),
        use_container_width=True
    )

    # 5️⃣ Plot future forecast
    st.subheader("Future Forecast")
    st.plotly_chart(
        plot_future(daily, res.future_mean, low=res.future_low, high=res.future_high,
                    title=f"Future {H}-day Forecast"),
        use_container_width=True
    )

    # 6️⃣ Download button
    csv = res.future_mean.rename("forecast").to_frame().to_csv().encode("utf-8")
    st.download_button("⬇️ Download forecast CSV", csv,
                       file_name=f"forecast_{model_name.replace(' ','_')}_{H}d.csv",
                       mime="text/csv")

except Exception as e:
    st.error(f"Something went wrong while fitting {model_name}: {e}")

st.markdown("---")
st.caption("Try changing horizon, changepoint flexibility, or seasonality mode. Prophet + Weather often performs best.")
