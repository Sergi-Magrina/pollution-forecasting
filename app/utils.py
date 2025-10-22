from __future__ import annotations
import pandas as pd
import numpy as np
from typing import Tuple, Optional, Dict
from sklearn.metrics import mean_absolute_error, mean_squared_error

def load_timeseries(df: pd.DataFrame, date_col: str, value_col: str, freq: Optional[str] = None) -> pd.Series:
    s = df[[date_col, value_col]].dropna()
    s[date_col] = pd.to_datetime(s[date_col], errors="coerce")
    s = s.dropna(subset=[date_col])
    s = s.set_index(date_col)[value_col].astype(float).sort_index()
    if freq:
        s = s.asfreq(freq)
    return s

def resample_and_fill(s: pd.Series, rule: str = "D") -> pd.Series:
    s = s.resample(rule).mean()
    # fill missing with forward fill then back fill as fallback
    return s.ffill().bfill()

def rolling_features(s: pd.Series, windows=(7, 30)) -> pd.DataFrame:
    df = pd.DataFrame({"value": s})
    for w in windows:
        df[f"roll_mean_{w}"] = s.rolling(w).mean()
    return df

def ts_train_test_split(s: pd.Series, test_size: int) -> Tuple[pd.Series, pd.Series]:
    return s.iloc[:-test_size], s.iloc[-test_size:]

def mape(y_true, y_pred) -> float:
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    mask = y_true != 0
    return (np.fabs((y_true[mask] - y_pred[mask]) / y_true[mask]).mean()) * 100

def evaluate_forecast(y_true, y_pred) -> Dict[str, float]:
    mae = mean_absolute_error(y_true, y_pred)
    rmse = mean_squared_error(y_true, y_pred, squared=False)
    mape_v = mape(y_true, y_pred)
    return {"MAE": mae, "RMSE": rmse, "MAPE": mape_v}