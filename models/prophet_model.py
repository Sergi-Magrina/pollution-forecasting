# models/prophet_model.py
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error
from .base import ForecastResult, mape
from prophet import Prophet

class ProphetModel:
    def __init__(self, cps=0.2, seasonal_mode="additive", regressors: pd.DataFrame | None = None):
        self.cps = float(cps)
        self.seasonal_mode = seasonal_mode
        self.regressors = regressors  # daily DataFrame indexed by datetime (same frequency as y)
        self.model_ = None
        self.used_regs_ = []

    def _align_regressors(self, idx: pd.DatetimeIndex) -> pd.DataFrame | None:
        """Return regressors aligned to idx (fill small gaps), or None if no regressors."""
        if self.regressors is None:
            return None
        reg = self.regressors.copy()
        reg.index = pd.to_datetime(reg.index)
        reg = reg.reindex(idx).ffill().bfill()
        return reg

    def _prep_df(self, y: pd.Series) -> pd.DataFrame:
        """History frame for fitting: ds, y (+ aligned regressors if available)."""
        y = y.copy()
        y.index = pd.to_datetime(y.index)
        df = y.rename("y").to_frame()
        reg = self._align_regressors(y.index)
        if reg is not None:
            df = df.join(reg, how="left")
        df = df.reset_index().rename(columns={"index": "ds", "datetime": "ds"})
        return df

    def fit(self, y: pd.Series):
        m = Prophet(
            daily_seasonality=False,
            weekly_seasonality=True,
            yearly_seasonality=True,
            changepoint_prior_scale=self.cps,
            seasonality_mode=self.seasonal_mode,
        )
        self.used_regs_ = []
        if self.regressors is not None:
            for reg in ["TEMP", "DEWP", "PRES", "WSPM"]:
                if reg in self.regressors.columns:
                    m.add_regressor(reg)
                    self.used_regs_.append(reg)

        df = self._prep_df(y)
        cols = ["ds", "y"] + self.used_regs_
        m.fit(df[cols])
        self.model_ = m
        return self

    def _future_known(self, idx: pd.DatetimeIndex) -> pd.DataFrame:
        """Build a frame for prediction on KNOWN future dates (e.g., test window) using real regressors."""
        df = pd.DataFrame({"ds": pd.to_datetime(idx)})
        if self.used_regs_:
            reg = self._align_regressors(idx)
            # reg has same index; attach the used regressor columns
            for c in self.used_regs_:
                df[c] = reg[c].values
        return df

    def _future_unknown(self, last_hist_idx: pd.DatetimeIndex, steps: int) -> pd.DataFrame:
        """
        Build a frame for prediction on UNKNOWN future (true forecast) dates:
        use Prophet's future dates and hold regressors at their last known values.
        """
        future_all = self.model_.make_future_dataframe(periods=steps, freq="D")
        future_steps = future_all["ds"].tail(steps).reset_index(drop=True)
        df = pd.DataFrame({"ds": future_steps})
        if self.used_regs_:
            # last known values from the history end:
            last_reg = self._align_regressors(last_hist_idx)[self.used_regs_].iloc[-1:]
            tail = pd.DataFrame(
                np.repeat(last_reg.to_numpy(), steps, axis=0),
                columns=self.used_regs_
            )
            for c in self.used_regs_:
                df[c] = tail[c].values
        return df

    def evaluate_and_forecast(self, y: pd.Series, H: int) -> ForecastResult:
        y = y.copy()
        y.index = pd.to_datetime(y.index)
        train, test = y.iloc[:-H], y.iloc[-H:]

        # --- test-window prediction using REAL regressors ---
        self.fit(train)
        future_t = self._future_known(test.index)  # real weather on test dates
        fcst_t = self.model_.predict(future_t)
        pred_test = pd.Series(fcst_t["yhat"].values, index=test.index, name="forecast")

        metrics = {
            "MAE": float(mean_absolute_error(test, pred_test)),
            "RMSE": float(mean_squared_error(test, pred_test, squared=False)),
            "MAPE": mape(test.values, pred_test.values),
        }

        # --- true future forecast using last-known regressors ---
        self.fit(y)
        future_f = self._future_unknown(y.index, H)
        fcst_f = self.model_.predict(future_f)
        future_mean = pd.Series(fcst_f["yhat"].values,
                                index=pd.date_range(y.index[-1] + pd.Timedelta(days=1), periods=H, freq="D"),
                                name="forecast")
        low = pd.Series(fcst_f["yhat_lower"].values, index=future_mean.index) if "yhat_lower" in fcst_f else None
        high = pd.Series(fcst_f["yhat_upper"].values, index=future_mean.index) if "yhat_upper" in fcst_f else None

        info = f"Prophet(cps={self.cps}, mode={self.seasonal_mode})" \
               + (f" + regressors {self.used_regs_}" if self.used_regs_ else "")

        return ForecastResult(pred_test=pred_test,
                              future_mean=future_mean,
                              future_low=low,
                              future_high=high,
                              info=info,
                              metrics=metrics)
