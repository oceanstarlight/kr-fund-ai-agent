import pytest
import numpy as np
import pandas as pd

from sentinel import config
from sentinel.data.store import MarketData
from sentinel.data.universe import PRESETS, normalize
from sentinel.events.library import Event, find_analogs, load_events
from sentinel.quant import betas as betas_mod
from sentinel.quant.engines import portfolio_impact, realized
from sentinel.quant.scenario import analog_eligibility, analog_windows
from sentinel.quant.stats import weighted_quantile


def test_weighted_quantile_matches_unweighted_median():
    assert weighted_quantile([1, 2, 3], [1, 1, 1], 0.5) == 2.0
    assert weighted_quantile([1.0], [5.0], 0.25) == 1.0
    # Heavier weight pulls the median toward that value.
    assert weighted_quantile([0, 10], [9, 1], 0.5) < 5


def test_presets_sum_to_one():
    for p in PRESETS.values():
        assert abs(sum(p.weights.values()) - 1.0) < 1e-9
        assert normalize(p.weights) == normalize(p.weights)


def test_backfill_fills_pre_inception_from_proxy():
    days = pd.bdate_range("2010-01-01", periods=10)
    spy = pd.Series(np.linspace(100, 109, 10), index=days)
    late = pd.Series([np.nan] * 5 + list(np.linspace(50, 54, 5)), index=days)
    md = MarketData.from_raw({"SPY": spy, "PSP": late}, {}, None)
    # PSP backfills from SPY, so its pre-inception returns equal SPY's.
    r = md.prices.pct_change()
    assert np.allclose(r["PSP"].iloc[1:4], r["SPY"].iloc[1:4])
    assert md.is_backfilled("PSP", days[2])
    assert not md.is_backfilled("PSP", days[7])


def test_window_uses_prior_close_as_base(synthetic_md):
    md = synthetic_md
    sat = pd.Timestamp("2019-09-14")  # Saturday
    base, end = md.window(sat, 1)
    assert base == pd.Timestamp("2019-09-13")
    assert end == pd.Timestamp("2019-09-16")


def test_betas_recover_known_loadings(synthetic_md):
    b = betas_mod.estimate_betas(synthetic_md, pd.Timestamp("2015-01-01"), ["SPY", "EFA"])
    # Synthetic VIX loading is -0.2 per unit of the underlying shock; diff factor is scaled by 0.1.
    assert b.beta.loc["SPY", "vix"] < 0
    assert (b.n_obs >= 52).all()


def test_betas_do_not_look_ahead(synthetic_md):
    md = synthetic_md
    asof = pd.Timestamp("2012-06-29")
    b1 = betas_mod.estimate_betas(md, asof, ["SPY"])
    b2 = betas_mod.estimate_betas(md.truncate(asof), asof, ["SPY"])
    pd.testing.assert_frame_equal(b1.beta, b2.beta)


def test_analog_eligibility_excludes_overlapping_windows(synthetic_md):
    events = load_events()
    target = next(e for e in events if e.id == "2020-pandemic")
    base = synthetic_md.trading_day_before(target.t0)
    ok = analog_eligibility(synthetic_md, base, 20)
    oil_war = next(e for e in events if e.id == "2020-oil-price-war")
    assert not ok(oil_war)  # its 20-day window runs past the pandemic declaration
    analogs = find_analogs(target.tags(), events, base, 10, 0.0, exclude_id=target.id, eligible=ok)
    assert all(a.event.id != target.id for a in analogs)
    assert all(synthetic_md.window(a.event.t0, 20)[1] < base for a in analogs)


def test_portfolio_impact_and_realized_shapes(synthetic_md):
    md = synthetic_md
    w = PRESETS["balanced"].weights
    events = load_events()
    target = next(e for e in events if e.id == "2022-russia-ukraine")
    base = md.trading_day_before(target.t0)
    analogs = find_analogs(target.tags(), events, base, 5, 0.0, exclude_id=target.id,
                           eligible=analog_eligibility(md, base, 20))
    trunc = md.truncate(base)
    b = betas_mod.estimate_betas(trunc, base, list(w))
    imp = portfolio_impact(trunc, w, analog_windows(trunc, analogs, 5), b, 5)
    assert imp.n_analogs == len(analogs)
    assert imp.blend["q25"] <= imp.blend["q50"] <= imp.blend["q75"]
    assert abs(imp.contribution.sum() - (pd.Series(w) * imp.asset_blend_median).sum()) < 1e-12
    r = realized(md, w, target.t0, 5)
    assert r is not None and np.isfinite(r["portfolio"])


def test_event_library_is_consistent():
    events = load_events()
    ids = [e.id for e in events]
    assert len(ids) == len(set(ids))
    for e in events:
        assert isinstance(e, Event)
        assert e.t0 >= e.date - pd.Timedelta(days=1), e.id
        # Longer gaps only for documented market closures (e.g. NYSE shut after 9/11).
        assert (e.t0 - e.date).days <= 3 or "closed" in e.note, e.id
        assert e.t0.dayofweek < 5, e.id
        assert e.selection in ("curated", "gpr_spike", "official"), e.id
        if e.selection == "gpr_spike":
            assert "GPR spike" in e.note, e.id


def test_ridge_cv_matches_brute_force_loo(monkeypatch):
    rng = np.random.default_rng(1)
    X = rng.normal(size=(120, 4))
    y = X @ np.array([0.5, 0.0, 0.2, 0.0]) + rng.normal(size=120)
    grid = (0.001, 0.01, 0.1, 1.0, 10.0)

    def loo(a):
        err = []
        for i in range(len(y)):
            m = np.arange(len(y)) != i
            c, _ = betas_mod._ridge(X[m], y[m], a)
            err.append((y[i] - y[m].mean() - (X[i] - X[m].mean(0)) @ c) ** 2)
        return np.mean(err)

    assert betas_mod.select_alpha(X, y, grid) == min(grid, key=loo)


def test_ridge_cv_mode_records_alpha(synthetic_md, monkeypatch):
    monkeypatch.setattr("sentinel.config.RIDGE_ALPHA", "cv")
    b = betas_mod.estimate_betas(synthetic_md, pd.Timestamp("2015-01-01"), ["SPY", "EFA"])
    assert b.alpha.isin(config.RIDGE_ALPHAS).all()


def test_bootstrap_skill_interval():
    from sentinel.backtest import bootstrap_skill

    rng = np.random.default_rng(0)
    base = np.abs(rng.normal(0, 0.02, 200))
    lo, hi = bootstrap_skill(base * 0.5, base)  # model halves every error
    assert lo == pytest.approx(0.5) and hi == pytest.approx(0.5)
    noisy = np.abs(rng.normal(0, 0.02, 30))
    lo, hi = bootstrap_skill(noisy, np.abs(rng.normal(0, 0.02, 30)))
    assert lo < hi
    assert np.isnan(bootstrap_skill(noisy[:3], noisy[:3])[0])
