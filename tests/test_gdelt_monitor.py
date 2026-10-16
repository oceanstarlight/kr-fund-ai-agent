import pandas as pd

from sentinel import gdelt_monitor as g


def _hist(n_slots: int, share: float, country: str = "IR") -> pd.DataFrame:
    start = pd.Timestamp("2025-06-12 00:00")
    rows = [{"country": country, "conflict_mentions": 80.0, "material": 0.5, "goldstein": -2.0,
             "share": share * (1 + 0.05 * (i % 3)), "slot": start + i * g.SLOT} for i in range(n_slots)]
    return pd.DataFrame(rows)


def _latest(slot, share: float, mentions: float = 300.0, country: str = "IR") -> tuple[pd.DataFrame, pd.DataFrame]:
    agg = pd.DataFrame([{"country": country, "conflict_mentions": mentions, "material": 0.8,
                         "goldstein": -8.0, "share": share, "slot": slot}])
    raw = pd.DataFrame([{"actor1": "IRAN", "actor2": "ISRAEL", "root": "19", "quad": 4, "goldstein": -10.0,
                         "mentions": mentions, "country": country, "place": "Tehran", "added": "x",
                         "url": "https://example.com/a", "slot": slot}])
    return agg, raw


def test_requires_two_consecutive_slots_and_respects_cooldown():
    hist = _hist(100, 0.01)
    s1 = hist["slot"].max() + g.SLOT
    agg, raw = _latest(s1, 0.08)
    alerts, streak = g.detect(hist, agg, raw, {}, {})
    assert alerts == [] and streak == {"IR": 1}  # first spike slot only arms the streak
    hist2 = pd.concat([hist, agg])
    agg2, raw2 = _latest(s1 + g.SLOT, 0.09)
    alerts, streak = g.detect(hist2, agg2, raw2, {}, streak)
    assert [a.country for a in alerts] == ["IR"]
    assert "교전" in alerts[0].top_actions and "example.com" in alerts[0].sources
    # Within the cooldown window the same country does not alert again.
    alerts, _ = g.detect(hist2, agg2, raw2, {"IR": str(s1)}, streak)
    assert alerts == []


def test_ignores_small_volume_and_short_history():
    hist = _hist(100, 0.01)
    s1 = hist["slot"].max() + g.SLOT
    agg, raw = _latest(s1, 0.08, mentions=10)
    assert g.detect(hist, agg, raw, {}, {"IR": 1})[0] == []
    short = _hist(10, 0.01)
    agg, raw = _latest(short["slot"].max() + g.SLOT, 0.08)
    assert g.detect(short, agg, raw, {}, {"IR": 1})[0] == []


def test_alert_event_text_is_plain_english():
    hist = _hist(100, 0.01)
    s1 = hist["slot"].max() + g.SLOT
    agg, raw = _latest(s1, 0.08)
    alerts, _ = g.detect(hist, agg, raw, {}, {"IR": 1})
    assert "conflict-related coverage" in alerts[0].event_text()
