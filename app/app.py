import os
import numpy as np
import pandas as pd
import plotly.graph_objs as go
import streamlit as st

# Optional libs (guarded where used)
from sklearn.metrics import mean_absolute_error, mean_squared_error

# -------------------------
# Page config
# -------------------------
st.set_page_config(page_title="PM2.5 Forecasting Dashboard", layout="wide")

# -------------------------
# Utilities
# -------------------------
@st.cache_data(show_spinner=False)
def load_clean_series():
    """Load the cleaned daily PM2.5 series as a Pandas Series with DatetimeIndex."""
    df = pd.read_csv(os.path.join("data", "clean_pm25.csv"), parse_dates=["datetime"], index_col="datetime")
    # Accept both Series or 1-col DataFrame shapes
    if isinstance(df, pd.DataFrame):
        if df.shape[1] == 1:
            s = df.iloc[:, 0]
            s.name = "pm25"
        else:
            # If user saved multiple columns, pick first
            s = df.iloc[:, 0]
            s.name = s.name or "pm25"
    else:
        s = df
        s.name = "pm25"
    s = s.asfreq("D")  # ensure daily frequency
    return s

@st.cache_data(show_spinner=False)
def load_daily_weather():
    """
    Load daily-averaged weather variables from the original CSV (if present).
    Returns DataFrame indexed by date with columns among ['TEMP','DEWP','PRES','WSPM'].
    """
    raw_path = os.path.join("data", "BeijingPM20100101_20151231.csv")
    if not os.path.exists(raw_path):
        return None

    raw = pd.read_csv(raw_path, na_values=["NA"])
    raw["datetime"] = pd.to_datetime(dict(year=raw["year"], month=raw["month"], day=raw["day"], hour=raw["hour"]), errors="coerce")
    raw = raw.set_index("datetime").sort_index()

    # Detect which columns exist
    candidates = [c for c in ["TEMP", "DEWP", "PRES", "WSPM", "Iws"] if c in raw.columns]
    if not candidates:
        return None

    daily_weather = raw.resample("D")[candidates].mean().ffill().bfill()
    # Normalize naming: Iws -> WSPM if needed
    if "Iws" in daily_weather.columns and "WSPM" not in daily_weather.columns:
        daily_weather = daily_weather.rename(columns={"Iws": "WSPM"})
    # Keep a canonical set if available
    keep = [c for c in ["TEMP", "DEWP", "PRES", "WSPM"] if c in daily_weather.columns]
    return daily_weather[keep] if keep else daily_weather

def mape(y_true, y_pred):
    y_true, y_pred = np.asarray(y_true), np.asarray(y_pred)
    mask = y_true != 0
    if mask.sum() == 0:
        return np.nan
    return float(np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100)

def evaluate(y_true, y_pred):
    return {
        "MAE": float(mean_absolute_error(y_true, y_pred)),
        "RMSE": float(mean_squared_error(y_true, y_pred, squared=False)),
        "MAPE": mape(y_true, y_pred),
    }

def plot_series(history: pd.Series, test: pd.Series | None = None, preds: pd.Series | None = None, title=""):
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=history.index, y=history.values, name="History", mode="lines"))
    if test is not None:
        fig.add_trace(go.Scatter(x=test.index, y=test.values, name="Test (actual)", mode="lines"))
    if preds is not None:
        fig.add_trace(go.Scatter(x=preds.index, y=preds.values, name="Prediction", mode="lines"))
    fig.update_layout(height=420, title=title, xaxis_title="Date", yaxis_title="PM2.5 (µg/m³)", legend=dict(orientation="h"))
    return fig

def plot_future(history: pd.Series, future_mean: pd.Series, future_low: pd.Series | None = None, future_high: pd.Series | None = None, title="Future Forecast"):
    fig = go.Figure()
    # Only plot last 6 months of history for clarity
    cutoff = history.index[-180] if len(history) >= 180 else history.index[0]
    fig.add_trace(go.Scatter(x=history.index[history.index >= cutoff], y=history[history.index >= cutoff], name="History (last 6m)", mode="lines"))
    fig.add_trace(go.Scatter(x=future_mean.index, y=future_mean.values, name="Forecast", mode="lines"))
    if future_low is not None and future_high is not None:
        fig.add_trace(go.Scatter(
            x=list(future_mean.index) + list(future_mean.index[::-1]),
            y=list(future_high.values) + list(future_low.values[::-1]),
            fill="toself",
            fillcolor="rgba(0, 123, 255, 0.2)",
            line=dict(color="rgba(255,255,255,0)"),
            name="Confidence Interval",
            showlegend=True,
        ))
    fig.update_layout(height=420, title=title, xaxis_title="Date", yaxis_title="PM2.5 (µg/m³)", legend=dict(orientation="h"))
    return fig


# -------------------------
# Models
# -------------------------
def forecast_arima(train: pd.Series, steps: int, seasonal: bool, m: int, use_log: bool):
    """
    Fit pmdarima.auto_arima on train and predict `steps`.
    Returns (pred_test: pd.Series, model_info:str)
    """
    import pmdarima as pm

    y = train.copy()
    if use_log:
        y = np.log1p(y)

    model = pm.auto_arima(
        y,
        start_p=0, start_q=0, max_p=5, max_q=5,
        d=None,
        seasonal=seasonal, m=(m if seasonal else 1),
        stepwise=True, suppress_warnings=True,
        information_criterion="aic",
        sarimax_kwargs={"enforce_stationarity": True, "enforce_invertibility": True},
        error_action="ignore",
    )
    preds = model.predict(n_periods=steps)
    if use_log:
        preds = np.expm1(preds)
    idx = pd.date_range(train.index[-1] + pd.Timedelta(days=1), periods=steps, freq="D")
    pred_series = pd.Series(preds, index=idx, name="forecast")
    order_txt = getattr(model, "order_", None)
    sorder_txt = getattr(model, "seasonal_order_", None)
    info = f"auto_arima order={order_txt}, seasonal_order={sorder_txt}"
    return pred_series, info

def forecast_prophet(train: pd.Series, steps: int, add_weather: bool, daily_weather: pd.DataFrame | None, cps: float = 0.2, seasonal_mode: str = "additive"):
    """
    Fit Prophet on train (with optional weather regressors) and predict `steps`.
    Returns (future_mean: pd.Series, future_low: pd.Series|None, future_high: pd.Series|None, info:str)
    """
    try:
        from prophet import Prophet
        from prophet.models import StanBackendEnum
        backend_kwargs = {"stan_backend": StanBackendEnum.CMDSTANPY}  # safe on cloud; ignored locally if not needed
    except Exception:
        from prophet import Prophet
        backend_kwargs = {}

    df_train = train.rename("y").to_frame().reset_index().rename(columns={"datetime": "ds"})
    m = Prophet(
        daily_seasonality=False,
        weekly_seasonality=True,
        yearly_seasonality=True,
        changepoint_prior_scale=cps,
        seasonality_mode=seasonal_mode,
        **backend_kwargs
    )
    used_regs = []
    if add_weather and (daily_weather is not None):
        # join on ds
        merged = df_train.set_index("ds").join(daily_weather, how="left").reset_index()
        df_train = merged
        for reg in ["TEMP", "DEWP", "PRES", "WSPM"]:
            if reg in df_train.columns:
                m.add_regressor(reg)
                used_regs.append(reg)

    m.fit(df_train[["ds", "y"] + used_regs])

    # future frame for steps ahead
    future = m.make_future_dataframe(periods=steps, freq="D")
    if used_regs:
        # need future regressor values: simplest is "hold last known values"
        # (or you could forecast regressors separately)
        last_vals = df_train.iloc[-1:][used_regs]
        future_regs = pd.concat([df_train[["ds"] + used_regs], pd.DataFrame(
            np.repeat(last_vals.values, steps, axis=0), columns=used_regs)], ignore_index=True)
        future_regs["ds"] = future["ds"]
        fcst = m.predict(future_regs[["ds"] + used_regs])
    else:
        fcst = m.predict(future)

    future_slice = fcst.tail(steps).set_index("ds")
    mean = future_slice["yhat"].rename("forecast")
    low = future_slice.get("yhat_lower", None)
    high = future_slice.get("yhat_upper", None)
    info = f"Prophet(cps={cps}, mode={seasonal_mode})" + (f" + regressors {used_regs}" if used_regs else "")
    return mean, low, high, info


# -------------------------
# UI
# -------------------------
st.title("🌫️ PM2.5 Forecasting Dashboard")
st.caption("Explore, model, and forecast daily PM2.5 levels (Beijing).")

daily = load_clean_series()
weather_df = load_daily_weather()  # may be None if file not present

with st.sidebar:
    st.header("Controls")
    horizon = st.selectbox("Forecast horizon (days)", [7, 30, 60], index=1)
    model_name = st.selectbox(
        "Model",
        ["ARIMA (auto, non-seasonal)", "SARIMA (weekly m=7)", "Prophet (baseline)", "Prophet + Weather"],
        index=3 if weather_df is not None else 2
    )
    use_log = st.checkbox("ARIMA: log1p transform", value=False, help="Stabilize variance for ARIMA/SARIMA only.")
    cps = st.slider("Prophet: changepoint_prior_scale", 0.05, 1.0, 0.2, 0.05, help="Higher = more flexible trend")
    seasonal_mode = st.selectbox("Prophet: seasonality mode", ["additive", "multiplicative"], index=0)

    st.markdown("---")
    st.caption("Test-window: last N days of the series (same as horizon) are held out for evaluation.")

# Train-test split
H = horizon
assert len(daily) > H + 30, "Not enough history for chosen horizon."
train = daily.iloc[:-H]
test = daily.iloc[-H:]

# EDA plot
st.subheader("Historical Series")
st.plotly_chart(plot_series(daily, title="Daily PM2.5 (µg/m³)"), use_container_width=True)

# Fit & predict on the test window (for metrics), then forecast future H days from the full series
st.subheader("Model Fit & Evaluation")

pred_test = None
future_mean = None
future_low = None
future_high = None
model_info = ""

try:
    if model_name == "ARIMA (auto, non-seasonal)":
        # Predict next H from end of train to compare to test
        pred_test, model_info = forecast_arima(train, H, seasonal=False, m=1, use_log=use_log)
        # Align indices with test
        pred_test.index = test.index
        # Future forecast from full series
        future_mean, _info = forecast_arima(daily, H, seasonal=False, m=1, use_log=use_log)
        model_info += " | " + _info

    elif model_name == "SARIMA (weekly m=7)":
        pred_test, model_info = forecast_arima(train, H, seasonal=True, m=7, use_log=use_log)
        pred_test.index = test.index
        future_mean, _info = forecast_arima(daily, H, seasonal=True, m=7, use_log=use_log)
        model_info += " | " + _info

    elif model_name == "Prophet (baseline)":
        # test-window prediction: train Prophet on train, predict next H days
        mean_t, low_t, high_t, info_t = forecast_prophet(train, H, add_weather=False, daily_weather=None, cps=cps, seasonal_mode=seasonal_mode)
        pred_test = mean_t.copy()
        pred_test.index = test.index
        # future from full series
        mean_f, low_f, high_f, info_f = forecast_prophet(daily, H, add_weather=False, daily_weather=None, cps=cps, seasonal_mode=seasonal_mode)
        future_mean, future_low, future_high = mean_f, low_f, high_f
        model_info = info_f

    elif model_name == "Prophet + Weather":
        mean_t, low_t, high_t, info_t = forecast_prophet(train, H, add_weather=True, daily_weather=weather_df, cps=cps, seasonal_mode=seasonal_mode)
        pred_test = mean_t.copy()
        pred_test.index = test.index
        mean_f, low_f, high_f, info_f = forecast_prophet(daily, H, add_weather=True, daily_weather=weather_df, cps=cps, seasonal_mode=seasonal_mode)
        future_mean, future_low, future_high = mean_f, low_f, high_f
        model_info = info_f

    # Metrics
    metrics = evaluate(test.values, pred_test.values)
    c1, c2, c3 = st.columns(3)
    c1.metric("MAE", f"{metrics['MAE']:.1f}")
    c2.metric("RMSE", f"{metrics['RMSE']:.1f}")
    c3.metric("MAPE", f"{metrics['MAPE']:.1f}%" if metrics['MAPE'] == metrics['MAPE'] else "—")  # NaN-safe

    st.caption(f"Model: {model_info}")

    # Plot test overlay
    st.plotly_chart(
        plot_series(history=train, test=test, preds=pred_test, title="Test Window — Actual vs Prediction"),
        use_container_width=True
    )

    # Future forecast plot
    st.subheader("Future Forecast")
    if future_low is not None and future_high is not None:
        st.plotly_chart(
            plot_future(history=daily, future_mean=future_mean, future_low=future_low, future_high=future_high, title=f"Future {H}-day Forecast"),
            use_container_width=True
        )
    else:
        st.plotly_chart(
            plot_future(history=daily, future_mean=future_mean, title=f"Future {H}-day Forecast"),
            use_container_width=True
        )

    # Download
    csv = future_mean.rename("forecast").to_frame().to_csv(index=True).encode("utf-8")
    st.download_button("⬇️ Download forecast CSV", csv, file_name=f"forecast_{model_name.replace(' ','_').replace('(','').replace(')','')}_{H}d.csv", mime="text/csv")

except Exception as e:
    st.error(f"Something went wrong while fitting {model_name}: {e}")
    st.stop()

st.markdown("---")
st.caption("Tip: Prophet + Weather often performs best on PM2.5 due to strong meteorological drivers. Try changing horizon, changepoint flexibility, and seasonal mode.")
