"""Shock Map: geography math, figure structure and the live tracker (all offline)."""

import numpy as np
import pandas as pd
import pytest

from sentinel import geo, gdelt_monitor as g, shock_map
from sentinel.agents import llm
from sentinel.agents.pipeline import run
from sentinel.data.universe import ASSETS, PRESETS


def test_country_weights_sum_to_one_and_have_coordinates():
    for t in ASSETS:
        w = geo.country_weights(t)
        assert abs(sum(w.values()) - 1.0) < 1e-9, t
        assert all(c == "OTHER" or c in geo.COUNTRIES for c in w), t
    assert set(geo.FIPS_TO_ISO3) == set(g.WATCH)
    assert all(c in geo.COUNTRIES for c in geo.FIPS_TO_ISO3.values())


def test_country_pnl_adds_up_to_portfolio_pnl():
    w = PRESETS["sovereign"].weights
    r = pd.Series({t: -0.01 * (i + 1) for i, t in enumerate(w)})
    pnl = geo.country_pnl(w, r, aum=1e9)
    assert pnl.sum() == pytest.approx(float((pd.Series(w) * r).sum()) * 1e9)
    assert pnl["USA"] < 0 and "OTHER" in pnl


def test_replay_frames_keep_trace_layout(synthetic_md):
    st = run(synthetic_md, PRESETS["sovereign"].weights, mode="replay", event_id="2024-iran-israel")
    fig = shock_map.replay_map(st, synthetic_md, 1e9, "USD")
    assert [f.name for f in fig.frames] == [f"D+{d}" for d in range(1, 21)]
    assert {len(f.data) for f in fig.frames} == {len(fig.data)}


def test_ask_map_uses_region_point(synthetic_md):
    st = run(synthetic_md, PRESETS["balanced"].weights, mode="ask", event_text="Iran closes the Strait of Hormuz")
    assert shock_map.event_origin(st)[:2] == shock_map.REGION_POINT["middle_east"]
    assert shock_map.signal_strength(st) == 0.0
    assert len(shock_map.static_map(st, 1e9, "USD").data) > 0


def test_library_rows_and_map(synthetic_md):
    rows = shock_map.library_rows(synthetic_md, PRESETS["balanced"].weights)
    assert rows["realized"].notna().sum() >= 20
    fig = shock_map.library_map(rows)
    ids = {i for t in fig.data if t.customdata is not None for i in t.customdata}
    assert ids == set(rows.dropna(subset=["lat"])["id"])


def _slot(t, shares):
    return pd.DataFrame([{"country": c, "conflict_mentions": m, "material": 0.5, "goldstein": -5.0,
                          "share": s, "slot": t} for c, (s, m) in shares.items()])


def test_country_scores_flags_spike_and_live_map_statuses():
    t0 = pd.Timestamp("2026-01-01")
    rng = np.random.default_rng(0)
    hist = pd.concat([_slot(t0 + i * g.SLOT, {"IR": (0.01 + rng.normal(0, 0.001), 100), "JA": (0.002, 10)})
                      for i in range(90)], ignore_index=True)
    latest = _slot(t0 + 90 * g.SLOT, {"IR": (0.05, 400), "JA": (0.02, 20)})
    sc = g.country_scores(hist, latest).set_index("country")
    assert sc.loc["IR", "z"] > g.GDELT_Z and sc.loc["JA", "z"] > g.GDELT_Z
    assert len(sc) == len(g.WATCH) and sc.loc["KS", "share"] == 0.0
    alert = g.GdeltAlert("IR", "이란", str(latest["slot"].iloc[0]), 0.05, 9.0, 400, [], [], [])
    fig = shock_map.live_map(sc.reset_index(), [alert], {"IR": pd.Series({"USA": -1e6, "JPN": 2e5})})
    names = {t.name for t in fig.data if t.name}
    assert {"경보", "급등(보도량 적음)", "평상"} <= names


def test_country_scores_wait_for_history():
    t0 = pd.Timestamp("2026-01-01")
    hist = pd.concat([_slot(t0 + i * g.SLOT, {"IR": (0.01, 100)}) for i in range(10)], ignore_index=True)
    sc = g.country_scores(hist, _slot(t0 + 10 * g.SLOT, {"IR": (0.05, 400)}))
    assert sc["z"].isna().all()


def test_thread_local_llm_switch(monkeypatch):
    monkeypatch.delenv("SENTINEL_DISABLE_LLM", raising=False)
    monkeypatch.setenv("GEMINI_API_KEY", "fake")
    assert llm.provider() == "gemini"
    with llm.disabled():
        assert llm.provider() == "none"
    assert llm.provider() == "gemini"


def test_no_llm_state_skips_llm_even_with_key(synthetic_md, monkeypatch):
    monkeypatch.delenv("SENTINEL_DISABLE_LLM", raising=False)
    monkeypatch.setenv("GEMINI_API_KEY", "fake")

    def boom(*_a):
        raise AssertionError("LLM must not be called")

    monkeypatch.setattr(llm, "_structured_gemini", boom)
    st = run(synthetic_md, PRESETS["balanced"].weights, mode="ask", event_text="Iran strikes", no_llm=True)
    assert st["classifier"] == "rules" and st["brief_engine"] == "template"
