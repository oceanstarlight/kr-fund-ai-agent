import pandas as pd

from sentinel.agents import llm
from sentinel.agents.pipeline import run
from sentinel.data.universe import PRESETS


def test_replay_runs_end_to_end(synthetic_md):
    st = run(synthetic_md, PRESETS["sovereign"].weights, mode="replay", event_id="2024-iran-israel", aum=1e9)
    nodes = [s["node"] for s in st["trace"]]
    assert nodes[0] == "orchestrator" and nodes[-1] == "evaluate"
    assert "verifier" in nodes and "brief" in nodes
    assert st["base"] < pd.Timestamp("2024-04-15")
    assert st["brief_engine"] == "template"
    assert set(st["realized"]) == {1, 5, 20}


def test_verifier_retry_loop_is_bounded(synthetic_md):
    # The earliest events have almost no eligible analogs, forcing retries.
    st = run(synthetic_md, PRESETS["balanced"].weights, mode="replay", event_id="2011-fukushima")
    nodes = [s["node"] for s in st["trace"]]
    assert nodes.count("analyst") == st["verification"]["attempts"]
    assert nodes.count("analyst") <= 3
    assert st["verification"]["grade"] == "C"


def test_ask_mode_uses_rule_classifier_without_llm(synthetic_md):
    st = run(synthetic_md, PRESETS["balanced"].weights, mode="ask",
             event_text="Iran closes the Strait of Hormuz after missile strikes")
    assert st["classifier"] == "rules"
    assert st["classification"]["region"] == "middle_east"
    assert "oil" in st["classification"]["channels"]
    assert "sentinel" not in [s["node"] for s in st["trace"]]
    assert "realized" not in st


def test_monitor_mode_stops_when_quiet(synthetic_md, monkeypatch):
    monkeypatch.setattr("sentinel.config.SENTINEL_Z_MARKET", 99.0)
    st = run(synthetic_md, PRESETS["balanced"].weights, mode="monitor")
    assert [s["node"] for s in st["trace"]] == ["orchestrator", "sentinel"]


def test_brief_audit_rejects_free_numbers():
    facts = {"port_q50_5d": "-1.20%", "event_name": "X"}
    assert llm.audit_brief("[[event_name]] 영향 D+5 [[port_q50_5d]]", facts) == []
    assert llm.audit_brief("손실은 약 3% 예상", facts)
    assert llm.audit_brief("[[unknown_key]]", facts)
    assert llm.fill_placeholders("D+5 [[port_q50_5d]]", facts) == "D+5 -1.20%"


def test_rule_classifier_korean():
    c = llm.classify_rules("대만 해협 봉쇄로 반도체 공급망 우려")
    assert c.region == "east_asia"
    assert "supply_chain" in c.channels
