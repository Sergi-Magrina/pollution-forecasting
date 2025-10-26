from dataclasses import dataclass
import numpy as np

def mape(y_true, y_pred):
    y_true, y_pred = np.asarray(y_true), np.asarray(y_pred)
    mask = y_true != 0
    return float(np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100) if mask.any() else float("nan")

@dataclass
class ForecastResult:
    pred_test: "pd.Series" | None
    future_mean: "pd.Series"
    future_low: "pd.Series | None" = None
    future_high: "pd.Series | None" = None
    info: str = ""
    metrics: dict | None = None
