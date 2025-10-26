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
        self.regressors = regressors  # daily DataFrame indexed by datetime or None
        self.model_ = None
        self.used_regs_ = []

    def _prep_df(self, y: pd.Series) -> pd.DataFrame:
        df = y.rename("y").to_frame().reset_index().rename(columns={"datetime": "ds"})
        if self.regressors is not None:
            # join weather on the same daily index
            reg = self.regressors.copy()
            reg = reg.loc[~reg.index.duplicated(keep="last")]
            df = df.set_index("ds").join(reg, how="left").reset_index()
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

    def _future_df(self, y: pd.Series, steps: int) -> pd.DataFrame:
        df_hist = self._prep_df(y)
        future = self.model_.make_future_dataframe(periods=steps, freq="D")
        if self.used_regs_:
            # hold last known regressor values constant (simple, robust)
            last_vals = df_hist.iloc[-1:][self.used_regs_]
            tail = pd.DataFrame(np.repeat(last_vals.values, steps, axis=0),
                                columns=self.used_regs_)
            tail.insert(0, "ds", future["ds"].values)
            return tail[["ds"] + self.used_regs_]
        return future

    def evaluate_and_forecast(self, y: pd.Series, H: int) -> ForecastResult:
        train, test = y.iloc[:-H], y.iloc[-H:]

        # test-window prediction
        self.fit(train)
        future_t = self._future_df(train, H)
        fcst_t = self.model_.predict(future_t)
        pred_test = pd.Series(fcst_t.tail(H)["yhat"].values, index=test.index, name="forecast")

        metrics = {
            "MAE": float(mean_absolute_error(test, pred_test)),
            "RMSE": float(mean_squared_error(test, pred_test, squared=False)),
            "MAPE": mape(test.values, pred_test.values),
        }

        # future forecast from full history
        self.fit(y)
        future_f = self._future_df(y, H)
        fcst_f = self.model_.predict(future_f).tail(H).set_index("ds")
        future_mean = fcst_f["yhat"].rename("forecast")
        low = fcst_f.get("yhat_lower", None)
        high = fcst_f.get("yhat_upper", None)

        info = f"Prophet(cps={self.cps}, mode={self.seasonal_mode})" \
               + (f" + regressors {self.used_regs_}" if self.used_regs_ else "")

        return ForecastResult(pred_test=pred_test,
                              future_mean=future_mean,
                              future_low=low,
                              future_high=high,
                              info=info,
                              metrics=metrics)
