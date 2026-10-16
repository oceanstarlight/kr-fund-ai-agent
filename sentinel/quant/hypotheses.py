"""Detection-time update: several hypotheses scored against the first-day market reaction.

At the close of the first reaction session (t0, when the sentinel can detect the shock) the
first-day factor moves are already observed. Each hypothesis (an interpretation of the event:
the planner's classification, an oil-supply shock, a broad risk-off, a rates/dollar shock, a
contained event) brings its own analogs. Analogs whose own first-day factor moves resemble the
observed ones get more weight, so the hypotheses the market reaction supports drive the
forecast of the REMAINING return (t0 close -> D+h close).

The weighting formula is fixed in advance (no tuning on the backtest):
    z_k  = (observed_k - scale * analog_k) / daily std_k        (k = factors with data)
    d^2  = mean_k z_k^2,   likelihood = exp(-d^2 / 2)
    analog weight  ~ sum over hypotheses of prior_h * similarity weight_(a|h) * likelihood_a
    posterior_h    ~ prior_h * sum_a similarity weight_(a|h) * likelihood_a
Uniform priors over hypotheses that found analogs.
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

MAX_HYPOTHESES = 4
SD_LOOKBACK = 252

# Transmission archetypes. Region is taken from the base classification.
ARCHETYPES: list[tuple[str, str, str, list[str]]] = [
    ("oil", "원유 공급 충격", "energy_supply", ["oil", "supply_chain"]),
    ("risk_off", "위험 회피 확산", "financial_stress", ["risk_off", "safe_haven", "banking"]),
    ("rates_usd", "금리·달러 충격", "financial_stress", ["rates", "usd", "em_fx"]),
    ("contained", "제한적 영향(빠른 진정)", "political", ["risk_off"]),
]


@dataclass
class Hypothesis:
    key: str
    label: str
    query: dict
    analogs: list[Analog] = field(default_factory=list)
    prior: float = 0.0
    likelihood: float = float("nan")  # similarity-weighted mean likelihood of its analogs
    posterior: float = 0.0


@dataclass
class DetectionResult:
    base: pd.Timestamp  # pre-event close
    detect: pd.Timestamp  # close of the first reaction session (forecast time)
    method: str
    observed_factors: pd.Series  # first-day factor moves
    observed_portfolio: float  # first-day portfolio return
    hypotheses: list[Hypothesis]
    pool: pd.DataFrame  # analog -> similarity, likelihood, distance, weight, hypotheses
    remaining: dict[int, HorizonImpact]  # horizon h -> forecast of t0 close -> D+h close
    cumulative: dict[int, dict[str, float]]  # horizon h -> q25/q50/q75 from the pre-event base
    notes: list[str] = field(default_factory=list)

    def top(self) -> Hypothesis | None:
        live = [h for h in self.hypotheses if h.analogs]
        return max(live, key=lambda h: h.posterior) if live else None


def _jaccard(a, b) -> float:
    a, b = set(a), set(b)
    return len(a & b) / len(a | b) if a | b else 1.0


def generate_hypotheses(classification: dict) -> list[Hypothesis]:
    """The base interpretation plus up to three archetypes that differ from it."""
    base = {k: classification[k] for k in ("type", "region", "channels")}
    out = [Hypothesis("base", "기본 해석(분류 결과)", base)]
    for key, label, typ, channels in ARCHETYPES:
        if len(out) >= MAX_HYPOTHESES:
            break
        if typ == base["type"] and _jaccard(channels, base["channels"]) >= 0.5:
            continue  # same reading as the base interpretation
        out.append(Hypothesis(key, label, {"type": typ, "region": base["region"], "channels": channels}))
    return out


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
    method: str = "hypotheses",
    vol_method: str = config.VOL_METHOD,
    interval_method: str = config.INTERVAL_METHOD,
    k: int = config.ANALOG_K,
    min_similarity: float = config.ANALOG_MIN_SIMILARITY,
) -> DetectionResult | None:
    """Forecast at the close of the first reaction session.

    method: "hypotheses" (all hypotheses, fingerprint weights), "base_fp" (base hypothesis only,
    fingerprint weights), "base" (base hypothesis only, similarity weights: the pre-event method
    applied at detection time).
    """
    w1 = md.window(t0, 1)
    if w1 is None:
        return None
    base, detect = w1
    vis = md.truncate(detect)
    obs_f = vis.factor_changes(base, detect)
    obs_p = float((pd.Series(weights) * vis.asset_returns(base, detect).reindex(list(weights))).sum())
    sd = factor_daily_sd(vis, base)
    # Analogs must have closed their longest window before the pre-event base, as in the main model.
    eligible = analog_eligibility(vis, base, max(config.HORIZONS))
    events = load_events()
    hyps = generate_hypotheses(classification) if method == "hypotheses" else generate_hypotheses(classification)[:1]
    for hp in hyps:
        hp.analogs = find_analogs(hp.query, events, base, k, min_similarity, exclude_id, False, eligible)
    live = [hp for hp in hyps if hp.analogs]
    notes = []
    if not live:
        return DetectionResult(base, detect, method, obs_f, obs_p, hyps, pd.DataFrame(), {}, {}, ["유사 사건 없음"])
    for hp in live:
        hp.prior = 1.0 / len(live)

    use_fp = method in ("hypotheses", "base_fp")
    rows: dict[str, dict] = {}
    for hp in live:
        lik_sum = 0.0
        for a in hp.analogs:
            r = rows.get(a.event.id)
            if r is None:
                aw = vis.window(a.event.t0, 1)
                f_a = vis.factor_changes(*aw) if aw else pd.Series(dtype=float)
                sc = vol_scale(vis, weights, aw[0], base, vol_method, 1) if aw else 1.0
                d2 = fingerprint_distance(obs_f, f_a, sd, sc) if use_fp else float("nan")
                lik = float(np.exp(-d2 / 2)) if np.isfinite(d2) else 1.0
                r = rows[a.event.id] = {"analog": a, "name": a.event.name, "similarity": a.similarity,
                                        "distance": d2, "likelihood": lik, "weight": 0.0, "hypotheses": []}
            r["weight"] += hp.prior * a.weight * r["likelihood"]
            r["hypotheses"].append(hp.label)
            lik_sum += a.weight * r["likelihood"]
        hp.likelihood = lik_sum
    z = sum(hp.prior * hp.likelihood for hp in live) or 1.0
    for hp in live:
        hp.posterior = hp.prior * hp.likelihood / z
    total = sum(r["weight"] for r in rows.values()) or 1.0
    for r in rows.values():
        r["weight"] /= total
    if use_fp and max(r["likelihood"] for r in rows.values()) < 0.05:
        notes.append("첫날 반응과 비슷한 과거 사건이 없음")

    betas = estimate_betas(vis, detect, list(weights))
    remaining, cumulative = {}, {}
    for h in horizons:
        if h <= 1:
            continue
        wins = [w for w in (_remaining_window(vis, r["analog"], h, r["weight"]) for r in rows.values()) if w]
        if not wins:
            continue
        p = portfolio_impact(vis, weights, wins, betas, h - 1, True, vol_method, interval_method)
        remaining[h] = p
        cumulative[h] = {q: (1 + obs_p) * (1 + p.blend[q]) - 1 for q in ("q25", "q50", "q75")}
    pool = pd.DataFrame([{k2: v for k2, v in r.items() if k2 != "analog"} | {"id": i} for i, r in rows.items()])
    pool = pool.sort_values("weight", ascending=False).reset_index(drop=True)
    return DetectionResult(base, detect, method, obs_f, obs_p, hyps, pool, remaining, cumulative, notes)


def realized_remaining(md: MarketData, weights: dict[str, float], t0: pd.Timestamp, h: int) -> float | None:
    """Realized portfolio return from the first reaction close to D+h close."""
    w1, wh = md.window(t0, 1), md.window(t0, h)
    if w1 is None or wh is None:
        return None
    r = md.asset_returns(w1[1], wh[1]).reindex(list(weights))
    return float(np.nansum(pd.Series(weights) * r))
