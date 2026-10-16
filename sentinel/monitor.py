"""Sentinel anomaly detection on market factors and the daily geopolitical risk index."""

from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np
import pandas as pd

from sentinel import config
from sentinel.data.store import MarketData
from sentinel.data.universe import FACTORS

WATCH_ASSETS = {"SPY": "미국 주식", "EEM": "신흥국 주식", "EWY": "한국 주식", "GLD": "금"}


@dataclass
class Signal:
    source: str
    key: str
    name: str
    date: str
    value: float
    z: float
    triggered: bool

    def as_dict(self) -> dict:
        return asdict(self)


def _z(series: pd.Series, at: pd.Timestamp, lookback: int) -> tuple[float, float]:
    hist = series.loc[:at].dropna()
    if len(hist) < 30 or at not in hist.index:
        return float("nan"), float("nan")
    x = hist.loc[at]
    ref = hist.iloc[-(lookback + 1) : -1]
    sd = ref.std()
    if not sd or np.isnan(sd):
        return float(x), float("nan")
    return float(x), float((x - ref.mean()) / sd)


def market_signals(md: MarketData, session: pd.Timestamp) -> list[Signal]:
    session = pd.Timestamp(session)
    out = []
    for k, f in FACTORS.items():
        lv = md.factors[k]
        chg = np.log(lv / lv.shift(1)) if f.transform == "logret" else lv.diff()
        v, z = _z(chg, session, config.SENTINEL_LOOKBACK_DAYS)
        if not np.isnan(z):
            out.append(Signal("market", k, f.name, str(session.date()), v, z, abs(z) >= config.SENTINEL_Z_MARKET))
    for t, name in WATCH_ASSETS.items():
        if t not in md.prices:
            continue
        r = md.prices[t].pct_change(fill_method=None)
        v, z = _z(r, session, config.SENTINEL_LOOKBACK_DAYS)
        if not np.isnan(z):
            out.append(Signal("market", t, name, str(session.date()), v, z, abs(z) >= config.SENTINEL_Z_MARKET))
    return out


def gpr_signal(md: MarketData, start: pd.Timestamp, end: pd.Timestamp) -> Signal | None:
    """Max daily GPR z-score over calendar days [start, end] (covers weekend events)."""
    if md.gpr is None or "GPRD" not in md.gpr:
        return None
    lg = np.log(md.gpr["GPRD"].clip(lower=1e-6))
    days = lg.loc[pd.Timestamp(start) : pd.Timestamp(end)].index
    best = None
    for d in days:
        v, z = _z(lg, d, 365)
        if np.isnan(z):
            continue
        if best is None or z > best.z:
            best = Signal("gpr", "GPRD", "지정학 리스크 지수(일간)", str(d.date()), float(np.exp(v)), z, z >= config.SENTINEL_Z_GPR)
    return best


def detect(md: MarketData, session: pd.Timestamp, event_date: pd.Timestamp | None = None) -> dict:
    """Run all detectors for one session. ``event_date`` widens the GPR window to the event day."""
    session = pd.Timestamp(session)
    prev = md.trading_day_before(session) or session
    sigs = market_signals(md, session)
    g = gpr_signal(md, min(prev + pd.Timedelta(days=1), pd.Timestamp(event_date or session)), session)
    if g:
        sigs.append(g)
    sigs.sort(key=lambda s: -abs(s.z))
    triggered = [s for s in sigs if s.triggered]
    return {
        "session": str(session.date()),
        "triggered": bool(triggered),
        "signals": [s.as_dict() for s in sigs],
        "top": [s.as_dict() for s in (triggered or sigs)[:3]],
    }
