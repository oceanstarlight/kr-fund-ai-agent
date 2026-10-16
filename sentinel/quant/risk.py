"""Historical-simulation risk measures for context next to the scenario estimate."""

from __future__ import annotations

import numpy as np
import pandas as pd

from sentinel.data.store import MarketData


def historical_var(
    md: MarketData,
    weights: dict[str, float],
    asof: pd.Timestamp,
    horizon: int,
    lookback_days: int = 756,
    level: float = 0.95,
) -> dict[str, float]:
    """h-day VaR / CVaR from overlapping windows in the last ``lookback_days`` sessions."""
    px = md.prices.loc[:asof, list(weights)].iloc[-(lookback_days + horizon) :]
    rets = (px / px.shift(horizon) - 1.0).dropna(how="all")
    w = pd.Series(weights)
    port = (rets.fillna(0.0) * w).sum(axis=1).to_numpy()
    if port.size < 30:
        return {"var": float("nan"), "cvar": float("nan"), "n": int(port.size)}
    cut = np.quantile(port, 1 - level)
    tail = port[port <= cut]
    return {"var": float(-cut), "cvar": float(-tail.mean()), "n": int(port.size)}
