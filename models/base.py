# models/base.py
from dataclasses import dataclass
from typing import Optional, Dict
import numpy as np
import pandas as pd

def mape(y_true, y_pred) -> float:
    y_true, y_pred = np.asarray(y_true), np.asarray(y_pred)
    mask = y_true != 0
    if mask.sum() == 0:
        return float("nan")
    return float(np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100)

@dataclass
class ForecastResult:
    pred_test: Optional[pd.Series]
    future_mean: pd.Series
    future_low: Optional[pd.Series] = None
    future_high: Optional[pd.Series] = None
    info: str = ""
    metrics: Optional[Dict[str, float]] = None
