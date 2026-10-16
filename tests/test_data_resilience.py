"""A failing data source must be reported, not crash loading or analysis."""

import numpy as np
import pandas as pd
import pytest

from sentinel import config
from sentinel.agents.pipeline import run
from sentinel.data import cache
from sentinel.data.store import DataUnavailable, MarketData
from sentinel.data.universe import ASSETS, PRESETS


@pytest.fixture
def fake_sources(monkeypatch, tmp_path):
    """Offline fake fetchers; individual sources can be switched to fail."""
    monkeypatch.setattr(config, "CACHE_DIR", tmp_path / "cache")
    monkeypatch.setattr(config, "MANIFEST_PATH", tmp_path / "cache" / "manifest.json")
    monkeypatch.setattr(cache, "SNAPSHOT_DIR", tmp_path / "snapshot")
    days = pd.bdate_range("2004-01-01", "2026-06-30")
    rng = np.random.default_rng(1)
    failing: set[str] = set()

    def make(name):
        def fetch(key):
            if name in failing or key in failing:
                raise RuntimeError(f"{name} down")
            s = pd.Series(100 * np.cumprod(1 + rng.normal(0, 0.01, len(days))), index=days, name=key)
            return s, f"https://example/{name}/{key}"
        return fetch

    monkeypatch.setattr(cache, "FETCHERS", {k: (make(k), "") for k in ("yahoo", "eia", "fed", "cboe")})
    monkeypatch.setattr(cache, "get_gpr", lambda *a, **k: (None, cache.SeriesStatus("gpr", "daily", "failed", "off")))
    return failing


def test_failed_sources_are_reported_and_analysis_continues(fake_sources):
    fake_sources.update({"eia", "PSP"})  # oil factor and one asset fail with no cache
    md = MarketData.load(offline=False)
    failed = {(r.source, r.key) for r in md.problems if r.origin == "failed"}
    assert ("eia", "RBRTE") in failed and ("yahoo", "PSP") in failed
    assert "oil" not in md.factors and "PSP" not in md.prices
    st = run(md, PRESETS["sovereign"].weights, mode="replay", event_id="2022-russia-ukraine")
    assert "PSP" not in st["weights"] and abs(sum(st["weights"].values()) - 1) < 1e-9
    assert "PSP" in st["trace"][0]["summary"]  # orchestrator reports the dropped asset
    assert st["brief"]


def test_network_failure_falls_back_to_snapshot_then_stale_cache(fake_sources):
    md = MarketData.load(offline=False)  # fills the cache
    assert all(r.origin == "network" for r in md.report if r.source != "gpr")
    assert cache.write_snapshot("eia", "RBRTE")
    fake_sources.update({"eia", "yahoo"})
    s, st = cache.fetch_series("eia", "RBRTE", max_age_hours=0)
    assert st.origin == "snapshot" and s is not None and "eia down" in st.error
    s, st = cache.fetch_series("yahoo", "SPY", max_age_hours=0)
    assert st.origin == "stale-cache" and s is not None


def test_missing_calendar_raises_with_report(fake_sources):
    fake_sources.add("SPY")
    with pytest.raises(DataUnavailable) as exc:
        MarketData.load(offline=False)
    assert any(r.key == "SPY" and r.origin == "failed" for r in exc.value.report)


def test_snapshot_only_for_public_domain_sources(fake_sources):
    MarketData.load(offline=False)
    assert not cache.write_snapshot("yahoo", "SPY")
    assert not cache.write_snapshot("cboe", "VIX")
    assert ASSETS  # universe untouched
