"""Synthetic market data so tests run offline and deterministically."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from sentinel.data.store import MarketData
from sentinel.data.universe import ASSETS, FACTORS


@pytest.fixture(autouse=True)
def _no_llm(monkeypatch):
    monkeypatch.setenv("SENTINEL_DISABLE_LLM", "1")


@pytest.fixture(scope="session")
def synthetic_md() -> MarketData:
    rng = np.random.default_rng(0)
    days = pd.bdate_range("2004-01-01", "2026-06-30")
    n = len(days)
    f_shocks = {k: rng.normal(0, 0.01, n) for k in FACTORS}
    factors = {}
    for k, f in FACTORS.items():
        if f.transform == "logret":
            factors[f.fred_id] = pd.Series(100 * np.exp(np.cumsum(f_shocks[k])), index=days)
        else:
            factors[f.fred_id] = pd.Series(3 + np.cumsum(f_shocks[k]) * 0.1, index=days)
    # Assets load on factors with known betas plus noise.
    prices = {}
    for i, t in enumerate(a for a in ASSETS if not ASSETS[a].synthetic):
        beta_oil = 0.1 * ((i % 3) - 1)
        r = beta_oil * f_shocks["oil"] - 0.2 * f_shocks["vix"] + rng.normal(0, 0.005, n)
        prices[t] = pd.Series(100 * np.cumprod(1 + r), index=days)
    raw_factors = {k: factors[f.fred_id] for k, f in FACTORS.items()}
    return MarketData.from_raw(prices, raw_factors, None)
