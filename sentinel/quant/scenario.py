"""Scenario construction from historical analog windows."""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from sentinel.data.store import MarketData
from sentinel.data.universe import FACTORS
from sentinel.events.library import Analog, Event
from sentinel.quant.stats import quantiles


@dataclass
class AnalogWindow:
    analog: Analog
    base: pd.Timestamp
    end: pd.Timestamp
    factor_change: pd.Series
    asset_return: pd.Series


def analog_eligibility(md: MarketData, target_base: pd.Timestamp, horizon: int):
    """Predicate: the analog's full ``horizon`` window closed before the forecast base date."""
    target_base = pd.Timestamp(target_base)

    def ok(ev: Event) -> bool:
        w = md.window(ev.t0, horizon)
        return w is not None and w[1] < target_base

    return ok


def analog_windows(md: MarketData, analogs: list[Analog], horizon: int) -> list[AnalogWindow]:
    out = []
    for a in analogs:
        w = md.window(a.event.t0, horizon)
        if w is None:
            continue
        base, end = w
        out.append(AnalogWindow(a, base, end, md.factor_changes(base, end), md.asset_returns(base, end)))
    return out


def scenario_table(windows: list[AnalogWindow]) -> pd.DataFrame:
    """Weighted quantiles of factor moves across analogs (display + engine A inputs)."""
    if not windows:
        return pd.DataFrame()
    w = [x.analog.weight for x in windows]
    rows = {}
    for k in FACTORS:
        vals = [x.factor_change.get(k, float("nan")) for x in windows]
        rows[k] = quantiles(vals, w)
    df = pd.DataFrame(rows).T
    df["name"] = [FACTORS[k].name for k in df.index]
    df["unit"] = [FACTORS[k].unit for k in df.index]
    return df
