import numpy as np
import pandas as pd

from sentinel.data.universe import PRESETS, normalize
from sentinel.events.library import load_events
from sentinel.quant import reaction


def _event(eid="2020-pandemic"):
    return next(e for e in load_events() if e.id == eid)


def test_fingerprint_distance_zero_for_identical_moves():
    obs = pd.Series({"oil": 0.02, "vix": 0.1})
    sd = pd.Series({"oil": 0.01, "vix": 0.05})
    assert reaction.fingerprint_distance(obs, obs, sd) == 0.0
    assert reaction.fingerprint_distance(obs, obs * 0, sd) == np.mean([4.0, 4.0])
    assert np.isnan(reaction.fingerprint_distance(obs, pd.Series(dtype=float), sd))


def test_detection_forecast_no_look_ahead_and_weights(synthetic_md):
    ev = _event()
    w, _ = synthetic_md.usable_weights(normalize(PRESETS["sovereign"].weights))
    r = reaction.detection_forecast(synthetic_md, w, ev.t0, ev.tags(), ev.id, (5, 20))
    assert r.detect == synthetic_md.trading_day_on_or_after(ev.t0)
    assert abs(r.pool["weight"].sum() - 1) < 1e-9
    assert ev.id not in set(r.pool["id"])
    for eid in r.pool["id"]:
        e = next(x for x in load_events() if x.id == eid)
        assert synthetic_md.window(e.t0, 20)[1] < r.base
    # Prices after the detection close must not change the forecast.
    md2 = synthetic_md.truncate(synthetic_md.window(ev.t0, 1)[1] + pd.Timedelta(days=400))
    r2 = reaction.detection_forecast(md2, w, ev.t0, ev.tags(), ev.id, (5, 20))
    assert r2.remaining[5].blend["q50"] == r.remaining[5].blend["q50"]
    # The base method keeps the library similarity weights (likelihood 1).
    rb = reaction.detection_forecast(synthetic_md, w, ev.t0, ev.tags(), ev.id, (5,), method="base")
    assert (rb.pool["likelihood"] == 1.0).all()
    assert np.allclose(rb.pool["weight"], rb.pool["similarity"] / rb.pool["similarity"].sum())


def test_realized_remaining_starts_after_first_day(synthetic_md):
    ev = _event()
    d1, d5 = synthetic_md.window(ev.t0, 1)[1], synthetic_md.window(ev.t0, 5)[1]
    p = synthetic_md.prices["SPY"]
    assert abs(reaction.realized_remaining(synthetic_md, {"SPY": 1.0}, ev.t0, 5) - (p[d5] / p[d1] - 1)) < 1e-12


def test_reaction_tool_unknowns_are_digit_free(synthetic_md):
    from sentinel import tools

    ev = _event()
    r = tools.reaction_update(synthetic_md, {"SPY": 1.0}, ev.t0, ev.tags(), ev.id)
    assert r.value is not None and r.unknowns
    assert not any(ch.isdigit() for u in r.unknowns for ch in u.message)
