import numpy as np
import pandas as pd
import math

def required_agents_workload(volume: float, aht_seconds: float, interval_minutes: int,
                             occupancy_cap: float, shrinkage: float) -> float:
    interval_seconds = interval_minutes * 60
    base = (volume * aht_seconds) / max(interval_seconds * occupancy_cap, 1e-9)
    return base / max(1.0 - shrinkage, 1e-6)

def erlang_c_probability_wait(a: float, n: int) -> float:
    if n <= a:
        return 1.0
    sum_terms = 0.0
    term = 1.0
    for k in range(0, n):
        if k > 0:
            term *= a / k
        sum_terms += term
    last = term * (a / n)
    denom = sum_terms + last * (n / (n - a))
    p0 = 1.0 / denom
    return last * (n / (n - a)) * p0

def service_level_erlang_c(volume: float, aht_seconds: float, interval_minutes: int,
                           agents: int, target_answer_seconds: int) -> float:
    interval_seconds = interval_minutes * 60
    lam = volume / max(interval_seconds, 1e-9)
    a = lam * aht_seconds
    if a <= 0:
        return 1.0
    pw = erlang_c_probability_wait(a, agents)
    exp_term = math.exp(-(agents - a) * (target_answer_seconds / aht_seconds))
    sl = 1.0 - pw * exp_term
    return max(0.0, min(1.0, sl))

def required_agents_erlang(volume: float, aht_seconds: float, interval_minutes: int,
                           target_sl: float, target_answer_seconds: int,
                           shrinkage: float, max_agents: int = 2000) -> int:
    interval_seconds = interval_minutes * 60
    lam = volume / max(interval_seconds, 1e-9)
    a = lam * aht_seconds
    if a <= 0:
        return 0

    start = max(1, int(math.ceil(a)))
    best = None
    for n in range(start, min(max_agents, start + 2000)):
        sl = service_level_erlang_c(volume, aht_seconds, interval_minutes, n, target_answer_seconds)
        if sl >= target_sl:
            best = n
            break
    if best is None:
        best = min(max_agents, start + 2000)

    return int(math.ceil(best / max(1.0 - shrinkage, 1e-6)))

def staffing_table(forecast_df: pd.DataFrame, interval_minutes: int,
                   method: str, aht_seconds: float, occupancy_cap: float,
                   shrinkage: float, target_sl: float, target_answer_seconds: int) -> pd.DataFrame:
    out = forecast_df.copy()
    if method == "Workload":
        out["required_agents"] = out["yhat"].apply(
            lambda v: required_agents_workload(v, aht_seconds, interval_minutes, occupancy_cap, shrinkage)
        )
        out["required_agents"] = np.ceil(out["required_agents"]).astype(int)
    else:
        out["required_agents"] = out["yhat"].apply(
            lambda v: required_agents_erlang(v, aht_seconds, interval_minutes, target_sl, target_answer_seconds, shrinkage)
        ).astype(int)
    return out