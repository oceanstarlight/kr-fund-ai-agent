"""Volatility forecasts (GARCH(1,1) via ``arch``) for regime scaling and prediction intervals."""

from __future__ import annotations

import warnings
from functools import lru_cache

import numpy as np
import pandas as pd

from sentinel.data.store import MarketData

LOOKBACK_DAYS = 756


def portfolio_returns(md: MarketData, weights: dict[str, float], asof: pd.Timestamp, days: int = LOOKBACK_DAYS) -> pd.Series:
    r = md.prices.loc[:asof, list(weights)].pct_change(fill_method=None).iloc[-days:]
    return (r.fillna(0.0) * pd.Series(weights)).sum(axis=1)


@lru_cache(maxsize=4096)
def _garch_variances(values: tuple[float, ...], horizon: int) -> tuple[float, ...]:
    """Per-day variance forecasts (fraction^2) for the next ``horizon`` days; EWMA fallback."""
    r = np.asarray(values) * 100.0  # percent scale keeps the optimizer well conditioned
    try:
        from arch import arch_model

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            res = arch_model(r, mean="Zero", vol="GARCH", p=1, q=1, dist="normal", rescale=False).fit(disp="off")
        var = res.forecast(horizon=horizon, reindex=False).variance.values[-1]
        if not np.all(np.isfinite(var)) or np.any(var <= 0):
            raise ValueError("bad forecast")
    except Exception:  # noqa: BLE001 - fall back to RiskMetrics EWMA
        lam, v = 0.94, float(np.var(r[:20])) if len(r) >= 20 else float(np.var(r))
        for x in r:
            v = lam * v + (1 - lam) * x * x
        var = np.full(horizon, v)
    return tuple(float(x) / 1e4 for x in var)


def garch_sigma(md: MarketData, weights: dict[str, float], asof: pd.Timestamp, horizon: int) -> float:
    """Forecast standard deviation of the portfolio's cumulative return over ``horizon`` days."""
    r = portfolio_returns(md, weights, asof)
    if len(r) < 120:
        return float("nan")
    var = _garch_variances(tuple(np.round(r.to_numpy(), 10)), max(horizon, 1))
    return float(np.sqrt(sum(var[:horizon])))
