import streamlit as st
import pandas as pd
import numpy as np
from datetime import timedelta
from utils import load_timeseries, resample_and_fill, rolling_features, ts_train_test_split, evaluate_forecast

st.set_page_config(page_title="Air Pollution Forecasting", layout="wide")

st.title("🌫️ Air Pollution Time-Series Forecasting Dashboard")
st.caption("Analyze PM2.5 trends and forecast future levels using ARIMA or Prophet.")

# Sidebar controls
with st.sidebar:
    st.header("⚙️ Controls")
    horizon = st.selectbox("Forecast horizon (days)", [7, 30, 60], index=0)
    model_type = st.radio("Model", ["ARIMA", "Prophet"], index=0)
    resample_rule = st.selectbox("Resample frequency", ["D", "W", "M"], index=0)
    test_size = st.number_input("Test size (days)", min_value=7, max_value=120, value=30, step=1)

st.subheader("1) Data")
uploaded = st.file_uploader("Upload a CSV with a datetime column and a PM2.5 column", type=["csv"])

if uploaded is not None:
    df = pd.read_csv(uploaded)
    st.write("Preview:", df.head())
    date_col = st.selectbox("Select datetime column", df.columns.tolist())
    value_col = st.selectbox("Select PM2.5 column", df.columns.tolist())
else:
    st.info("No file uploaded yet. You can also place a CSV at `data/clean_pm25.csv`.")
    try:
        df = pd.read_csv("data/clean_pm25.csv")
        st.success("Loaded `data/clean_pm25.csv`.")
        st.write("Preview:", df.head())
        date_col = st.selectbox("Select datetime column", df.columns.tolist(), index=0)
        value_col = st.selectbox("Select PM2.5 column", df.columns.tolist(), index=1 if len(df.columns)>1 else 0)
    except Exception:
        df = None
        date_col = value_col = None

if df is not None and date_col and value_col:
    s = load_timeseries(df, date_col, value_col)
    s = resample_and_fill(s, rule=resample_rule)
    st.line_chart(s, height=220)

    st.subheader("2) Rolling Averages")
    feats = rolling_features(s, windows=(7, 30))
    st.line_chart(feats, height=220)

    # Decomposition (optional simple moving averages already shown; full STL left to notebooks)

    st.subheader("3) Train/Test & Forecast")
    if len(s) > (test_size + horizon + 5):
        train, test = ts_train_test_split(s, test_size=test_size)
        st.write(f"Train range: {train.index.min().date()} → {train.index.max().date()}")
        st.write(f"Test range: {test.index.min().date()} → {test.index.max().date()}")

        # Fit models
        if model_type == "ARIMA":
            try:
                import pmdarima as pm
                with st.spinner("Fitting auto-ARIMA..."):
                    arima = pm.auto_arima(train, seasonal=False, suppress_warnings=True, stepwise=True)
                pred_test = pd.Series(arima.predict(n_periods=len(test)), index=test.index)
                metrics = evaluate_forecast(test, pred_test)

                # Future forecast
                future = arima.predict(n_periods=horizon)
                future_index = pd.date_range(start=s.index[-1] + pd.Timedelta(days=1), periods=horizon, freq='D')
                forecast = pd.Series(future, index=future_index)

            except Exception as e:
                st.error(f"ARIMA failed: {e}")
                metrics, forecast = None, None

        else:  # Prophet
            try:
                from prophet import Prophet
                dfp = train.reset_index()
                dfp.columns = ["ds", "y"]
                m = Prophet(daily_seasonality=True, weekly_seasonality=True, yearly_seasonality=True)
                m.fit(dfp)
                future_df = m.make_future_dataframe(periods=len(test) + horizon, freq='D')
                forecast_full = m.predict(future_df).set_index("ds")["yhat"]
                pred_test = forecast_full.iloc[-(len(test)+horizon):].iloc[:len(test)]
                metrics = evaluate_forecast(test, pred_test)

                forecast = forecast_full.iloc[-horizon:]
            except Exception as e:
                st.error(f"Prophet failed: {e}")
                metrics, forecast = None, None

        if metrics:
            st.write("**Metrics on Test Set:**", metrics)

        if forecast is not None:
            st.subheader("4) Forecast")
            st.line_chart(pd.DataFrame({"history": s, "forecast": forecast}), height=240)
            st.download_button("Download Forecast CSV", forecast.to_frame("forecast").to_csv().encode("utf-8"), file_name="forecast.csv")

    else:
        st.warning("Not enough data after resampling for the chosen test size and horizon.")
else:
    st.stop()