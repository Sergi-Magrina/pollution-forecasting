import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error
from .base import ForecastResult, mape
import pmdarima as pm

class AutoARIMA:
    def __init__(self, seasonal=False, m=1, use_log=False):
        self.seasonal = seasonal
        self.m = m
        self.use_log = use_log
        self.model_ = None

    def _maybe_log(self, y):
        return np.log1p(y) if self.use_log else y

    def _maybe_exp(self, yhat):
        return np.expm1(yhat) if self.use_log else yhat

    def fit(self, y: pd.Series):
        y_ = self._maybe_log(y)
        self.model_ = pm.auto_arima(
            y_,
            start_p=0, start_q=0, max_p=5, max_q=5, d=None,
            seasonal=self.seasonal, m=(self.m if self.seasonal else 1),
            stepwise=True, suppress_warnings=True, information_criterion="aic",
            sarimax_kwargs={"enforce_stationarity": True, "enforce_invertibility": True},
            error_action="ignore",
        )
        return self

    def predict(self, steps: int, index: pd.DatetimeIndex):
        yhat = self.model_.predict(n_periods=steps)
        yhat = self._maybe_exp(yhat)
        return pd.Series(yhat, index=index, name="forecast")

    def evaluate_and_forecast(self, y: pd.Series, H: int):
        train, test = y.iloc[:-H], y.iloc[-H:]
        self.fit(train)
        pred_test = self.predict(H, test.index)
        # Metrics
        metrics = {
            "MAE": float(mean_absolute_error(test, pred_test)),
            "RMSE": float(mean_squared_error(test, pred_test, squared=False)),
            "MAPE": mape(test.values, pred_test.values),
        }
        # Future from full series
        future_idx = pd.date_range(y.index[-1] + pd.Timedelta(days=1), periods=H, freq="D")
        self.fit(y)  # refit on all data
        future_mean = self.predict(H, future_idx)
        info = f"auto_arima order={getattr(self.model_, 'order_', None)}, seasonal_order={getattr(self.model_, 'seasonal_order_', None)}"
        return ForecastResult(pred_test=pred_test, future_mean=future_mean, info=info, metrics=metrics)
