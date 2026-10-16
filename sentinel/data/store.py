"""Aligned market data panel used by every quant tool."""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from sentinel.data import cache, quality
from sentinel.data.universe import ASSETS, FACTORS

CALENDAR_TICKER = "SPY"


class DataUnavailable(RuntimeError):
    """Raised only when the trading calendar itself (SPY) cannot be loaded."""

    def __init__(self, message: str, report: list | None = None):
        super().__init__(message)
        self.report = report or []


@dataclass
class MarketData:
    prices: pd.DataFrame  # total-return index levels on the trading calendar
    factors: pd.DataFrame  # factor levels on the trading calendar
    gpr: pd.DataFrame | None = None  # daily GPR (calendar days)
    inception: dict[str, pd.Timestamp] = field(default_factory=dict)
    report: list = field(default_factory=list)  # cache.SeriesStatus per source series
    quality: list = field(default_factory=list)  # quality.Failure per broken rule (raw series)

    @property
    def calendar(self) -> pd.DatetimeIndex:
        return self.prices.index

    @property
    def last_date(self) -> pd.Timestamp:
        return self.calendar[-1]

    @property
    def problems(self) -> list:
        """Series that failed or fell back to a snapshot/stale cache."""
        return [r for r in self.report if r.origin in ("failed", "snapshot", "stale-cache")]

    def usable_weights(self, weights: dict[str, float]) -> tuple[dict[str, float], list[str]]:
        """Drop assets without price data and renormalize, so analysis can still run."""
        ok = {t: w for t, w in weights.items() if t in self.prices and self.prices[t].notna().any()}
        dropped = [t for t in weights if t not in ok]
        total = sum(ok.values())
        return ({t: w / total for t, w in ok.items()} if total > 0 else {}), dropped

    # ------------------------------------------------------------------ build
    @classmethod
    def load(cls, offline: bool = False, max_age_hours: float | None = 24) -> "MarketData":
        """Fetch all sources in parallel. A failing source is reported, not raised.

        Yahoo requests stay sequential inside one worker (rate limit); the official factor
        sources and the GPR file each run in their own worker.
        """
        tickers = [t for t, a in ASSETS.items() if not a.synthetic]

        def yahoo_all():
            return [(t, *cache.fetch_series("yahoo", t, max_age_hours, offline)) for t in tickers]

        def factor(k):
            f = FACTORS[k]
            return (k, *cache.fetch_series(f.source, f.source_id, max_age_hours, offline))

        fac_keys = [k for k, f in FACTORS.items() if f.source != "derived"]
        with ThreadPoolExecutor(max_workers=len(fac_keys) + 2) as ex:
            fut_y = ex.submit(yahoo_all)
            fut_f = [ex.submit(factor, k) for k in fac_keys]
            fut_g = ex.submit(cache.get_gpr, max_age_hours, offline)
            y_res = fut_y.result()
            f_res = [f.result() for f in fut_f]
            gpr, gpr_status = fut_g.result()

        report = [st for _, _, st in y_res] + [st for _, _, st in f_res] + [gpr_status]
        raw = {t: s for t, s, _ in y_res if s is not None}
        fac = {k: s for k, s, _ in f_res if s is not None}
        if CALENDAR_TICKER not in raw:
            raise DataUnavailable(f"{CALENDAR_TICKER} price history is unavailable; cannot build the calendar", report)
        md = cls.from_raw(raw, fac, gpr)
        md.report = report
        # Validate the raw series as fetched; failures are reported, never raised.
        md.quality = quality.validate_all(raw, fac, gpr)
        return md

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
                dep_df = pd.DataFrame({d: rets[d] for d in deps})
                proxy = (dep_df.fillna(0.0) * pd.Series(deps)).sum(axis=1)
                # No proxy where none of the dependencies existed yet (keeps NaN, not 0%).
                proxy = proxy.where(dep_df.notna().any(axis=1))
                # Pre-inception days plus the inception day itself (its own return is NaN).
                mask = (rets[t].index <= inception[t]) & rets[t].isna()
                rets[t] = rets[t].where(~mask, proxy)
            if nxt == pending:
                break
            pending = nxt

        ret_df = pd.DataFrame(rets).iloc[1:]
        prices = (1.0 + ret_df.fillna(0.0)).cumprod() * 100.0
        # Keep NaN where neither the asset nor a proxy existed.
        prices = prices.where(ret_df.notna().cummax())
        factors = factors.loc[prices.index]
        for k, f in FACTORS.items():
            if f.source == "derived":
                num, den = f.source_id.split("/")
                if num in prices and den in prices:
                    factors[k] = prices[num] / prices[den]
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
