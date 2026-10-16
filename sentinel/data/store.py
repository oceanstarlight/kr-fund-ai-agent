"""Aligned market data panel used by every quant tool."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from sentinel.data import cache
from sentinel.data.universe import ASSETS, FACTORS

CALENDAR_TICKER = "SPY"


@dataclass
class MarketData:
    prices: pd.DataFrame  # total-return index levels on the trading calendar
    factors: pd.DataFrame  # factor levels on the trading calendar
    gpr: pd.DataFrame | None = None  # daily GPR (calendar days)
    inception: dict[str, pd.Timestamp] = field(default_factory=dict)

    @property
    def calendar(self) -> pd.DatetimeIndex:
        return self.prices.index

    @property
    def last_date(self) -> pd.Timestamp:
        return self.calendar[-1]

    # ------------------------------------------------------------------ build
    @classmethod
    def load(cls, offline: bool = False, max_age_hours: float | None = 24) -> "MarketData":
        raw = {}
        for t, a in ASSETS.items():
            if a.synthetic:
                continue
            raw[t] = cache.get_series("yahoo", t, max_age_hours, offline)
        fac = {k: cache.get_series("fred", f.fred_id, max_age_hours, offline) for k, f in FACTORS.items()}
        try:
            gpr = cache.get_gpr(max_age_hours, offline)
        except Exception:  # GPR is optional context for the sentinel
            gpr = None
        return cls.from_raw(raw, fac, gpr)

    @classmethod
    def from_raw(
        cls,
        raw_prices: dict[str, pd.Series],
        raw_factors: dict[str, pd.Series],
        gpr: pd.DataFrame | None = None,
    ) -> "MarketData":
        cal = raw_prices[CALENDAR_TICKER].dropna().index.sort_values()
        factors = pd.DataFrame(
            {k: s.sort_index().reindex(cal, method="ffill", limit=5) for k, s in raw_factors.items()},
            index=cal,
        )
        rets: dict[str, pd.Series] = {}
        inception: dict[str, pd.Timestamp] = {}
        for t, s in raw_prices.items():
            s = s.sort_index().reindex(cal).ffill(limit=5)
            first = s.first_valid_index()
            inception[t] = first if first is not None else cal[-1]
            rets[t] = s.pct_change(fill_method=None)
        if "krw" in factors:
            krw = factors["krw"]
            rets["KRW_CASH"] = krw.shift(1) / krw - 1.0
            inception["KRW_CASH"] = krw.first_valid_index() or cal[-1]

        # Backfill pre-inception returns from documented proxies, resolving chains.
        pending = [t for t, a in ASSETS.items() if a.backfill and t in rets]
        for _ in range(len(pending) + 1):
            if not pending:
                break
            nxt = []
            for t in pending:
                deps = ASSETS[t].backfill
                if any(d not in rets for d in deps) or any(d in pending and d != t for d in deps):
                    nxt.append(t)
                    continue
                proxy = sum(rets[d].fillna(0.0) * w for d, w in deps.items())
                mask = rets[t].index < inception[t]
                rets[t] = rets[t].where(~mask, proxy)
            if nxt == pending:
                break
            pending = nxt

        ret_df = pd.DataFrame(rets).iloc[1:]
        prices = (1.0 + ret_df.fillna(0.0)).cumprod() * 100.0
        # Keep NaN where neither the asset nor a proxy existed.
        prices = prices.where(ret_df.notna().cummax())
        factors = factors.loc[prices.index]
        return cls(prices=prices, factors=factors, gpr=gpr, inception=inception)

    # ---------------------------------------------------------------- helpers
    def truncate(self, asof: pd.Timestamp) -> "MarketData":
        """Data visible at the close of ``asof`` (no look-ahead)."""
        asof = pd.Timestamp(asof)
        gpr = self.gpr.loc[:asof] if self.gpr is not None else None
        return MarketData(self.prices.loc[:asof], self.factors.loc[:asof], gpr, self.inception)

    def trading_day_on_or_after(self, d: pd.Timestamp) -> pd.Timestamp | None:
        pos = self.calendar.searchsorted(pd.Timestamp(d), side="left")
        return self.calendar[pos] if pos < len(self.calendar) else None

    def trading_day_before(self, d: pd.Timestamp) -> pd.Timestamp | None:
        pos = self.calendar.searchsorted(pd.Timestamp(d), side="left") - 1
        return self.calendar[pos] if pos >= 0 else None

    def window(self, t0: pd.Timestamp, h: int) -> tuple[pd.Timestamp, pd.Timestamp] | None:
        """(base close, end close) for an h-trading-day window starting at session t0."""
        start = self.trading_day_on_or_after(t0)
        base = self.trading_day_before(t0)
        if start is None or base is None:
            return None
        pos = self.calendar.get_loc(start) + h - 1
        if pos >= len(self.calendar):
            return None
        return base, self.calendar[pos]

    def asset_returns(self, base: pd.Timestamp, end: pd.Timestamp) -> pd.Series:
        return self.prices.loc[end] / self.prices.loc[base] - 1.0

    def factor_changes(self, base: pd.Timestamp, end: pd.Timestamp) -> pd.Series:
        out = {}
        for k, f in FACTORS.items():
            if k not in self.factors:
                continue
            a, b = self.factors.at[base, k], self.factors.at[end, k]
            out[k] = np.log(b / a) if f.transform == "logret" else b - a
        return pd.Series(out)

    def is_backfilled(self, ticker: str, d: pd.Timestamp) -> bool:
        inc = self.inception.get(ticker)
        return inc is not None and pd.Timestamp(d) < inc
