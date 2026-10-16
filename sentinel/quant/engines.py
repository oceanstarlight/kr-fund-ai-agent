"""Engine A (factor model) and engine B (analog replay) portfolio impact estimates."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from sentinel.data.store import MarketData
from sentinel.quant.betas import BetaResult
from sentinel.quant.scenario import AnalogWindow
from sentinel.quant.stats import quantiles, weighted_quantile
from sentinel.quant.vol import garch_sigma

Z_QUARTILE = 0.6745  # standard normal 75th percentile


@dataclass
class HorizonImpact:
    horizon: int
    n_analogs: int
    engine_a: dict[str, float]  # portfolio return quantiles
    engine_b: dict[str, float]
    blend: dict[str, float]
    per_analog: pd.DataFrame  # analog_id x [A, B, blend, weight]
    asset_blend_median: pd.Series  # per-asset weighted median of blended return
    contribution: pd.Series  # weight * asset_blend_median
    backfilled_weight: float  # portfolio weight whose analog-window data came from proxies
    notes: list[str] = field(default_factory=list)
    interval_method: str = "analog"
    interval_sigma: float = float("nan")  # forecast std of the h-day portfolio return (garch)

    @property
    def engine_gap(self) -> float:
        return abs(self.engine_a["q50"] - self.engine_b["q50"])


def engine_a_asset(betas: BetaResult, factor_change: pd.Series) -> pd.Series:
    b = betas.beta.reindex(columns=factor_change.index)
    return b.fillna(0.0) @ factor_change.fillna(0.0)


def portfolio_vol(md: MarketData, weights: dict[str, float], asof: pd.Timestamp, days: int = 60) -> float:
    """Daily volatility of the static-weight portfolio over ``days`` sessions up to ``asof``."""
    r = md.prices.loc[:asof, list(weights)].pct_change(fill_method=None).iloc[-days:]
    port = (r.fillna(0.0) * pd.Series(weights)).sum(axis=1)
    return float(port.std()) if len(port) > 10 else float("nan")


def vol_scale(
    md: MarketData, weights: dict[str, float], analog_base: pd.Timestamp, target_base: pd.Timestamp,
    method: str = "garch", horizon: int = 1,
) -> float:
    """Rescale an analog's moves to today's volatility regime, clipped to [0.5, 2].

    "garch": ratio of GARCH(1,1) volatility forecasts over the same horizon at the two base
    dates (GARCH mean-reverts, so a one-day ratio would overstate long-horizon scaling).
    "realized": ratio of trailing 60-day realized volatilities.
    """
    if method == "garch":
        va, vt = garch_sigma(md, weights, analog_base, horizon), garch_sigma(md, weights, target_base, horizon)
    else:
        va, vt = portfolio_vol(md, weights, analog_base), portfolio_vol(md, weights, target_base)
    if not va or np.isnan(va) or np.isnan(vt):
        return 1.0
    return float(np.clip(vt / va, 0.5, 2.0))


def portfolio_impact(
    md: MarketData,
    weights: dict[str, float],
    windows: list[AnalogWindow],
    betas: BetaResult,
    horizon: int,
    use_vol_scale: bool = True,
    vol_method: str = "garch",
    interval_method: str = "garch",
) -> HorizonImpact:
    tickers = list(weights)
    w = pd.Series(weights)
    rows, asset_blend, notes = [], [], []
    backfilled = 0.0
    for x in windows:
        sc = vol_scale(md, weights, x.base, betas.asof, vol_method, horizon) if use_vol_scale else 1.0
        ra = engine_a_asset(betas, x.factor_change * sc).reindex(tickers)
        rb = x.asset_return.reindex(tickers) * sc
        missing_b = rb.isna()
        if missing_b.any():
            # Asset (and its proxy) did not exist yet: fall back to engine A for that asset.
            rb = rb.where(~missing_b, ra)
            notes.append(f"{x.analog.event.id}: 엔진B 데이터 없음 → 엔진A 대체 ({', '.join(rb.index[missing_b])})")
        bf = [t for t in tickers if md.is_backfilled(t, x.base)]
        backfilled = max(backfilled, float(w[bf].sum()) if bf else 0.0)
        blend_asset = 0.5 * (ra + rb)
        asset_blend.append(blend_asset.rename(x.analog.event.id))
        rows.append(
            {
                "analog": x.analog.event.id,
                "name": x.analog.event.name,
                "weight": x.analog.weight,
                "similarity": x.analog.similarity,
                "vol_scale": sc,
                "A": float((w * ra).sum()),
                "B": float((w * rb).sum()),
            }
        )
    per = pd.DataFrame(rows)
    if per.empty:
        nan = {k: float("nan") for k in ("q25", "q50", "q75", "min", "max")}
        return HorizonImpact(horizon, 0, nan, nan, nan, per, pd.Series(dtype=float), pd.Series(dtype=float), 0.0, ["유사 사건 없음"])
    per["blend"] = 0.5 * (per["A"] + per["B"])
    wts = per["weight"].to_numpy()
    ab = pd.DataFrame(asset_blend)
    med = pd.Series({t: weighted_quantile(ab[t].to_numpy(), wts, 0.5) for t in tickers})
    blend = quantiles(per["blend"].to_numpy(), wts)
    sigma = float("nan")
    if interval_method == "garch":
        # Prediction interval = analog spread plus the market noise expected over the horizon.
        # A few analogs alone understate uncertainty (backtest coverage was well below 50%).
        mu = float(np.average(per["blend"], weights=wts)) if wts.sum() > 0 else float(per["blend"].mean())
        sd_analog = float(np.sqrt(np.average((per["blend"] - mu) ** 2, weights=wts))) if len(per) > 1 else 0.0
        sigma = garch_sigma(md, weights, betas.asof, horizon)
        if np.isfinite(sigma):
            half = Z_QUARTILE * float(np.sqrt(sd_analog**2 + sigma**2))
            blend["q25"], blend["q75"] = blend["q50"] - half, blend["q50"] + half
    return HorizonImpact(
        horizon=horizon,
        n_analogs=len(per),
        engine_a=quantiles(per["A"].to_numpy(), wts),
        engine_b=quantiles(per["B"].to_numpy(), wts),
        blend=blend,
        per_analog=per,
        asset_blend_median=med,
        contribution=(w * med).sort_values(),
        backfilled_weight=backfilled,
        notes=notes,
        interval_method=interval_method if np.isfinite(sigma) or interval_method != "garch" else "analog",
        interval_sigma=sigma,
    )


def realized(md: MarketData, weights: dict[str, float], t0: pd.Timestamp, horizon: int) -> dict | None:
    """Realized portfolio return over the same window (None if not yet observable)."""
    win = md.window(t0, horizon)
    if win is None:
        return None
    base, end = win
    r = md.asset_returns(base, end).reindex(list(weights))
    w = pd.Series(weights)
    return {
        "base": base,
        "end": end,
        "portfolio": float(np.nansum(w * r)),
        "assets": r,
        "factors": md.factor_changes(base, end),
    }
