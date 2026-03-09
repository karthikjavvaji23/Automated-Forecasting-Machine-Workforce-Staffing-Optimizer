import numpy as np

def rmse(y_true, y_pred) -> float:
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    return float(np.sqrt(np.mean((y_true - y_pred) ** 2)))

def mape(y_true, y_pred) -> float:
    """
    Mean Absolute Percentage Error
    """
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)

    denom = np.where(y_true == 0, 1.0, y_true)

    return float(np.mean(np.abs((y_true - y_pred) / denom)) * 100.0)


def wmape(y_true, y_pred) -> float:
    """
    Weighted MAPE
    Much more stable for call center / operations forecasting
    """
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)

    denom = np.sum(np.abs(y_true))

    if denom == 0:
        return 0.0

    return float(np.sum(np.abs(y_true - y_pred)) / denom * 100.0)


def mae(y_true, y_pred) -> float:
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)

    return float(np.mean(np.abs(y_true - y_pred)))