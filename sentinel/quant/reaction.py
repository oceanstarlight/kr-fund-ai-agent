"""Detection-time update: re-weight analogs by how closely their first-day reaction matches.

At the close of the first reaction session (t0, when the sentinel can detect the shock) the
first-day factor moves are already observed. Analogs whose own first-day factor moves resemble
the observed ones get more weight, and the forecast covers the REMAINING return
(t0 close -> D+h close).

The weighting formula is fixed in advance (no tuning on the backtest):
    z_k  = (observed_k - scale * analog_k) / daily std_k        (k = factors with data)
    d^2  = mean_k z_k^2,   likelihood = exp(-d^2 / 2)
    analog weight ~ similarity weight * likelihood

Detection-time backtest: lower D+20 error than the similarity-weighted method, but no method
beat a zero forecast for the remaining return (see reports/detection_*.md).
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from sentinel import config
from sentinel.data.store import MarketData
from sentinel.data.universe import FACTORS
from sentinel.events.library import Analog, find_analogs, load_events
from sentinel.quant.betas import estimate_betas
from sentinel.quant.engines import HorizonImpact, portfolio_impact, vol_scale
from sentinel.quant.scenario import AnalogWindow, analog_eligibility

SD_LOOKBACK = 252
METHODS = ("base", "reaction")  # similarity weights only / similarity x first-day likelihood


@dataclass
class DetectionResult:
    base: pd.Timestamp  # pre-event close
    detect: pd.Timestamp  # close of the first reaction session (forecast time)
    method: str
    observed_factors: pd.Series  # first-day factor moves
    observed_portfolio: float  # first-day portfolio return
    pool: pd.DataFrame  # id, name, similarity, distance, likelihood, weight
    remaining: dict[int, HorizonImpact]  # horizon h -> forecast of t0 close -> D+h close
    cumulative: dict[int, dict[str, float]]  # horizon h -> q25/q50/q75 from the pre-event base
    notes: list[str] = field(default_factory=list)


def factor_daily_sd(md: MarketData, asof: pd.Timestamp, days: int = SD_LOOKBACK) -> pd.Series:
    """Std of one-day factor moves over ``days`` sessions up to ``asof`` (same transforms as the model)."""
    lv = md.factors.loc[:asof].iloc[-(days + 1):]
    out = {}
    for k, f in FACTORS.items():
        if k not in lv:
            continue
        chg = np.log(lv[k] / lv[k].shift(1)) if f.transform == "logret" else lv[k].diff()
        out[k] = float(chg.std())
    return pd.Series(out)


def fingerprint_distance(observed: pd.Series, analog: pd.Series, sd: pd.Series, scale: float = 1.0) -> float:
    """Mean squared standardized gap between first-day factor moves (NaN if nothing to compare)."""
    z = (observed - scale * analog.reindex(observed.index)) / sd.reindex(observed.index)
    z = z[np.isfinite(z)]
    return float((z**2).mean()) if len(z) else float("nan")


def _remaining_window(md: MarketData, a: Analog, h: int, weight: float) -> AnalogWindow | None:
    w1, wh = md.window(a.event.t0, 1), md.window(a.event.t0, h)
    if w1 is None or wh is None:
        return None
    start, end = w1[1], wh[1]
    return AnalogWindow(Analog(a.event, a.similarity, weight, a.reasons), start, end,
                        md.factor_changes(start, end), md.asset_returns(start, end))


def detection_forecast(
    md: MarketData,
    weights: dict[str, float],
    t0: pd.Timestamp,
    classification: dict,
    exclude_id: str | None = None,
    horizons=(5, 20),
    method: str = "reaction",
    vol_method: str = config.VOL_METHOD,
    interval_method: str = config.INTERVAL_METHOD,
    k: int = config.ANALOG_K,
    min_similarity: float = config.ANALOG_MIN_SIMILARITY,
) -> DetectionResult | None:
    """Forecast of the remaining return at the close of the first reaction session.

    method: "reaction" (similarity x first-day likelihood) or "base" (similarity weights only:
    the pre-event method applied at detection time, the backtest comparison).
    """
    w1 = md.window(t0, 1)
    if w1 is None:
        return None
    base, detect = w1
    vis = md.truncate(detect)
    obs_f = vis.factor_changes(base, detect)
    obs_p = float((pd.Series(weights) * vis.asset_returns(base, detect).reindex(list(weights))).sum())
    # Analogs must have closed their longest window before the pre-event base, as in the main model.
    eligible = analog_eligibility(vis, base, max(config.HORIZONS))
    analogs = find_analogs(classification, load_events(), base, k, min_similarity, exclude_id, False, eligible)
    if not analogs:
        return DetectionResult(base, detect, method, obs_f, obs_p, pd.DataFrame(), {}, {}, ["유사 사건 없음"])

    sd = factor_daily_sd(vis, base)
    rows = []
    for a in analogs:
        d2, lik = float("nan"), 1.0
        if method == "reaction":
            aw = vis.window(a.event.t0, 1)
            if aw:
                sc = vol_scale(vis, weights, aw[0], base, vol_method, 1)
                d2 = fingerprint_distance(obs_f, vis.factor_changes(*aw), sd, sc)
                lik = float(np.exp(-d2 / 2)) if np.isfinite(d2) else 1.0
        rows.append({"analog": a, "id": a.event.id, "name": a.event.name, "similarity": a.similarity,
                     "distance": d2, "likelihood": lik, "weight": a.weight * lik})
    total = sum(r["weight"] for r in rows) or 1.0
    for r in rows:
        r["weight"] /= total
    notes = []
    if method == "reaction" and max(r["likelihood"] for r in rows) < 0.05:
        notes.append("첫날 반응과 비슷한 과거 사건이 없음")

    betas = estimate_betas(vis, detect, list(weights))
    remaining, cumulative = {}, {}
    for h in horizons:
        if h <= 1:
            continue
        wins = [w for w in (_remaining_window(vis, r["analog"], h, r["weight"]) for r in rows) if w]
        if not wins:
            continue
        p = portfolio_impact(vis, weights, wins, betas, h - 1, True, vol_method, interval_method)
        remaining[h] = p
        cumulative[h] = {q: (1 + obs_p) * (1 + p.blend[q]) - 1 for q in ("q25", "q50", "q75")}
    pool = pd.DataFrame([{k2: v for k2, v in r.items() if k2 != "analog"} for r in rows])
    pool = pool.sort_values("weight", ascending=False).reset_index(drop=True)
    return DetectionResult(base, detect, method, obs_f, obs_p, pool, remaining, cumulative, notes)


def realized_remaining(md: MarketData, weights: dict[str, float], t0: pd.Timestamp, h: int) -> float | None:
    """Realized portfolio return from the first reaction close to D+h close."""
    w1, wh = md.window(t0, 1), md.window(t0, h)
    if w1 is None or wh is None:
        return None
    r = md.asset_returns(w1[1], wh[1]).reindex(list(weights))
    return float(np.nansum(pd.Series(weights) * r))
