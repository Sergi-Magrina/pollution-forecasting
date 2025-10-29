import os, sys
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

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

    # Align to PM series index (important)
    pm_index = load_clean_series().index
    daily_weather = daily_weather.reindex(pm_index).ffill().bfill()

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

# Cache the entire run so Prophet isn't refit on every rerun
@st.cache_data(show_spinner=True)
def run_model_cached(
    model_name: str,
    daily: pd.Series,
    weather_df: pd.DataFrame | None,
    H: int,
    use_log: bool,
    cps: float,
    seasonal_mode: str,
):
    # Instantiate the model (classes you already have)
    if model_name == "ARIMA (auto, non-seasonal)":
        from models.arima_model import AutoARIMA
        model = AutoARIMA(seasonal=False, m=1, use_log=use_log)
    elif model_name == "SARIMA (weekly m=7)":
        from models.arima_model import AutoARIMA
        model = AutoARIMA(seasonal=True, m=7, use_log=use_log)
    elif model_name == "Prophet (baseline)":
        from models.prophet_model import ProphetModel
        model = ProphetModel(cps=cps, seasonal_mode=seasonal_mode)
    else:
        from models.prophet_model import ProphetModel
        model = ProphetModel(cps=cps, seasonal_mode=seasonal_mode, regressors=weather_df)

    res = model.evaluate_and_forecast(daily, H)

    # Return plain-serializable parts (so cache works smoothly)
    out = {
        "metrics": res.metrics,
        "info": res.info,
        "pred_test": res.pred_test.to_frame().reset_index().values.tolist(),
        "future_mean": res.future_mean.to_frame().reset_index().values.tolist(),
        "future_low": res.future_low.to_frame().reset_index().values.tolist() if res.future_low is not None else None,
        "future_high": res.future_high.to_frame().reset_index().values.tolist() if res.future_high is not None else None,
    }
    return out


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
st.title("PM2.5 Forecasting Dashboard")
st.plotly_chart(plot_series(daily, title="Daily PM2.5 (µg/m³)"), use_container_width=True)

# ---------------------------------------------------
# MODEL FITTING AND FORECASTING
# ---------------------------------------------------
st.subheader("Model Fit & Evaluation")

# Add a form so reruns don't auto-trigger heavy fits
with st.form("run_form"):
    cols = st.columns([1,1,1,2])
    with cols[0]:
        st.write(f"Model: **{model_name}**")
    with cols[1]:
        st.write(f"Horizon: **{H} days**")
    with cols[2]:
        if "Prophet" in model_name:
            st.write(f"CPS: **{cps:.2f}**, Mode: **{seasonal_mode}**")
        else:
            st.write(f"ARIMA log1p: **{use_log}**")
    run_clicked = st.form_submit_button("🚀 Run forecast")

if not run_clicked:
    st.info("Set options in the sidebar and click **Run forecast**.")
    st.stop()

try:
    # Run (cached). Changing any input invalidates cache and recomputes once.
    packed = run_model_cached(
        model_name=model_name,
        daily=daily,
        weather_df=weather_df,
        H=H,
        use_log=use_log,
        cps=float(cps),
        seasonal_mode=seasonal_mode,
    )

    # Unpack cached payload back to Series for plotting
    pred_test_df = pd.DataFrame(packed["pred_test"], columns=["datetime","forecast"]).set_index("datetime")
    future_mean_df = pd.DataFrame(packed["future_mean"], columns=["datetime","forecast"]).set_index("datetime")
    future_low_df = pd.DataFrame(packed["future_low"], columns=["datetime","yhat_lower"]).set_index("datetime") if packed["future_low"] else None
    future_high_df = pd.DataFrame(packed["future_high"], columns=["datetime","yhat_upper"]).set_index("datetime") if packed["future_high"] else None

    pred_test = pred_test_df["forecast"]
    future_mean = future_mean_df["forecast"]
    future_low = future_low_df["yhat_lower"] if future_low_df is not None else None
    future_high = future_high_df["yhat_upper"] if future_high_df is not None else None

    metrics = packed["metrics"] or {}
    c1, c2, c3 = st.columns(3)
    c1.metric("MAE", f"{metrics.get('MAE', float('nan')):.1f}")
    c2.metric("RMSE", f"{metrics.get('RMSE', float('nan')):.1f}")
    mape_val = metrics.get("MAPE", float("nan"))
    c3.metric("MAPE", f"{mape_val:.1f}%" if mape_val == mape_val else "—")

    st.caption(f"Model: {packed['info']}")

    # Plot test overlay
    st.plotly_chart(
        plot_series(train, test, pred_test, title="Test Window — Actual vs Prediction"),
        use_container_width=True
    )

    # Future forecast plot
    st.subheader("Future Forecast")
    st.plotly_chart(
        plot_future(daily, future_mean, low=future_low, high=future_high, title=f"Future {H}-day Forecast"),
        use_container_width=True
    )

    # Download
    csv = future_mean.rename("forecast").to_frame().to_csv().encode("utf-8")
    st.download_button("⬇️ Download forecast CSV", csv,
        file_name=f"forecast_{model_name.replace(' ','_')}_{H}d.csv",
        mime="text/csv")

except Exception as e:
    st.error(f"Something went wrong while fitting {model_name}: {e}")


st.markdown("---")
st.caption("Try changing horizon, changepoint flexibility, or seasonality mode. Prophet + Weather often performs best.")
