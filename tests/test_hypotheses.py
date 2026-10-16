import numpy as np
import pandas as pd

from sentinel.data.universe import PRESETS, normalize
from sentinel.events.library import load_events
from sentinel.quant import hypotheses as hyp


def _event(eid="2020-pandemic"):
    return next(e for e in load_events() if e.id == eid)


def test_generate_hypotheses_keeps_region_and_skips_duplicate_reading():
    cls = {"type": "energy_supply", "region": "middle_east", "channels": ["oil", "supply_chain"]}
    hs = hyp.generate_hypotheses(cls)
    assert hs[0].key == "base" and len(hs) <= hyp.MAX_HYPOTHESES
    assert "oil" not in [h.key for h in hs]  # same reading as the base classification
    assert all(h.query["region"] == "middle_east" for h in hs)


def test_fingerprint_distance_zero_for_identical_moves():
    obs = pd.Series({"oil": 0.02, "vix": 0.1})
    sd = pd.Series({"oil": 0.01, "vix": 0.05})
    assert hyp.fingerprint_distance(obs, obs, sd) == 0.0
    assert hyp.fingerprint_distance(obs, obs * 0, sd) == np.mean([4.0, 4.0])
    assert np.isnan(hyp.fingerprint_distance(obs, pd.Series(dtype=float), sd))


def test_detection_forecast_no_look_ahead_and_weights(synthetic_md):
    ev = _event()
    w = normalize(PRESETS["sovereign"].weights)
    w, _ = synthetic_md.usable_weights(w)
    r = hyp.detection_forecast(synthetic_md, w, ev.t0, ev.tags(), ev.id, (5, 20))
    assert r.detect == synthetic_md.trading_day_on_or_after(ev.t0)
    assert abs(r.pool["weight"].sum() - 1) < 1e-9
    assert abs(sum(h.posterior for h in r.hypotheses) - 1) < 1e-9
    assert ev.id not in set(r.pool["id"])
    for a in (a for h in r.hypotheses for a in h.analogs):
        assert synthetic_md.window(a.event.t0, 20)[1] < r.base
    # Changing prices after the detection close must not change the forecast.
    md2 = synthetic_md.truncate(synthetic_md.window(ev.t0, 1)[1] + pd.Timedelta(days=400))
    r2 = hyp.detection_forecast(md2, w, ev.t0, ev.tags(), ev.id, (5, 20))
    assert r2.remaining[5].blend["q50"] == r.remaining[5].blend["q50"]
    # Base-only method with similarity weights reproduces the library similarity weights.
    rb = hyp.detection_forecast(synthetic_md, w, ev.t0, ev.tags(), ev.id, (5,), method="base")
    sims = {a.event.id: a.weight for a in rb.hypotheses[0].analogs}
    assert all(abs(row.weight - sims[row.id]) < 1e-9 for row in rb.pool.itertuples())


def test_realized_remaining_starts_after_first_day(synthetic_md):
    ev = _event()
    w = {"SPY": 1.0}
    d1, d5 = synthetic_md.window(ev.t0, 1)[1], synthetic_md.window(ev.t0, 5)[1]
    p = synthetic_md.prices["SPY"]
    assert abs(hyp.realized_remaining(synthetic_md, w, ev.t0, 5) - (p[d5] / p[d1] - 1)) < 1e-12


def test_reaction_tool_unknowns_are_digit_free(synthetic_md):
    from sentinel import tools

    ev = _event()
    r = tools.reaction_hypotheses(synthetic_md, {"SPY": 1.0}, ev.t0, ev.tags(), ev.id)
    assert r.value is not None and r.unknowns
    assert not any(ch.isdigit() for u in r.unknowns for ch in u.message)
