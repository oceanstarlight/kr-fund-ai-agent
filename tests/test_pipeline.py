import pandas as pd

from sentinel.agents import llm
from sentinel.agents.pipeline import run
from sentinel.data.universe import PRESETS
from sentinel.events.library import load_events


def test_replay_runs_end_to_end(synthetic_md):
    st = run(synthetic_md, PRESETS["sovereign"].weights, mode="replay", event_id="2024-iran-israel", aum=1e9)
    nodes = [s["node"] for s in st["trace"]]
    assert nodes[0] == "orchestrator" and nodes[-1] == "evaluate"
    assert {"planner", "executor", "verifier", "reviewer", "brief"} <= set(nodes)
    assert st["base"] < pd.Timestamp("2024-04-15")
    assert st["brief_engine"] == "template"
    assert set(st["realized"]) == {1, 5, 20}


def test_verifier_retry_loop_is_bounded(synthetic_md):
    # The earliest event inside the data has almost no eligible analogs, forcing retries.
    first = next(e.id for e in load_events() if synthetic_md.trading_day_before(e.t0) is not None)
    st = run(synthetic_md, PRESETS["balanced"].weights, mode="replay", event_id=first)
    nodes = [s["node"] for s in st["trace"]]
    assert nodes.count("planner") == st["verification"]["attempts"]
    assert nodes.count("planner") <= 3
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


def test_brief_audit_blocks_known_loopholes():
    facts = {"port_q50_5d": "-1.20%", "base_date": "2025-06-12"}
    for bad in ["손실은 삼 퍼센트 수준", "about twenty percent", "D+99 전망", "[[port_q50_5d]]% 하락", "손실이 두 배"]:
        assert llm.audit_brief(bad, facts), bad
    # Numbered lists and allowed horizons pass.
    assert llm.audit_brief("1. [[port_q50_5d]] (D+5)\n2. 기준일 [[base_date]]", facts) == []


def test_backfill_keeps_nan_before_any_proxy_exists():
    import numpy as np
    days = pd.bdate_range("2010-01-01", periods=10)
    spy = pd.Series(np.linspace(100, 109, 10), index=days)
    ief = pd.Series([np.nan] * 6 + [100, 101, 102, 103], index=days)
    from sentinel.data.store import MarketData
    md = MarketData.from_raw({"SPY": spy, "IEF": ief, "TLT": ief.copy()}, {}, None)
    # TLT backfills from IEF only, which did not exist yet: no fake flat prices.
    assert md.prices["TLT"].iloc[:4].isna().all()


def test_provider_selection(monkeypatch):
    monkeypatch.delenv("SENTINEL_DISABLE_LLM", raising=False)
    for k in ("GEMINI_API_KEY", "GOOGLE_API_KEY", "ANTHROPIC_API_KEY", "ANTHROPIC_AUTH_TOKEN"):
        monkeypatch.delenv(k, raising=False)
    assert llm.provider() == "none"
    monkeypatch.setenv("ANTHROPIC_API_KEY", "x")
    assert llm.provider() == "anthropic"
    monkeypatch.setenv("GEMINI_API_KEY", "y")
    assert llm.provider() == "gemini"
    monkeypatch.setattr("sentinel.config.LLM_PROVIDER", "anthropic")
    assert llm.provider() == "anthropic"
    monkeypatch.setenv("SENTINEL_DISABLE_LLM", "1")
    assert llm.provider() == "none"


def test_gemini_failure_falls_back_to_rules(monkeypatch):
    monkeypatch.delenv("SENTINEL_DISABLE_LLM", raising=False)
    monkeypatch.setenv("GEMINI_API_KEY", "invalid-key-for-test")

    def boom(*_args):
        raise RuntimeError("network down")

    monkeypatch.setattr(llm, "_structured_gemini", boom)
    plan, engine = llm.plan_event("Iran missile strike on Israel")
    assert engine.startswith("rules (gemini error")
    assert plan.classification.region == "middle_east"
    review, problems = llm.review_result({"x": "1"}, {})
    assert review is None and problems[0].startswith("gemini error")


def test_llm_planner_and_reviewer_path(synthetic_md, monkeypatch):
    """Simulated LLM: planner widens search on feedback, reviewer caution downgrades the grade."""
    monkeypatch.delenv("SENTINEL_DISABLE_LLM", raising=False)
    monkeypatch.setenv("GEMINI_API_KEY", "fake")
    calls = []

    def fake(schema, system, user, smart):
        calls.append(schema.__name__)
        if schema is llm.Plan:
            return llm.Plan(
                classification=llm.EventClassification(type="military_strike", region="middle_east",
                                                       channels=["oil", "risk_off"], headline_ko="h", rationale_ko="r"),
                candidates=[llm.SearchParams(analog_k=5, min_similarity=0.3, same_type_only=False, label_ko="좁은 검색"),
                            llm.SearchParams(analog_k=8, min_similarity=0.1, same_type_only=False, label_ko="넓은 검색")],
                rationale_ko="계획")
        return llm.Review(verdict="caution", concerns_ko=["유사 사건 수가 적음"],
                          interpretation_md="### 해석\n- D+5 기본 시나리오는 [[port_q50_5d]] 입니다.")

    monkeypatch.setattr(llm, "_structured_gemini", fake)
    st = run(synthetic_md, PRESETS["balanced"].weights, mode="ask", event_text="Iran strikes")
    assert calls[0] == "Plan" and calls[-1] == "Review"
    assert st["classifier"] == "llm:gemini"
    assert st["brief_engine"] == "llm"
    assert st["verification"]["grade"] != st["verification"]["rule_grade"] or st["verification"]["rule_grade"] == "C"
    assert "[[" not in st["brief"] and "### 검토 의견" in st["brief"]


def test_reviewer_text_with_free_numbers_is_rejected(synthetic_md, monkeypatch):
    monkeypatch.delenv("SENTINEL_DISABLE_LLM", raising=False)
    monkeypatch.setenv("GEMINI_API_KEY", "fake")

    def fake(schema, system, user, smart):
        if schema is llm.Plan:
            return llm.plan_rules("Iran strikes", 0)
        return llm.Review(verdict="confirm", concerns_ko=[], interpretation_md="### 해석\n- 손실은 약 3% 예상")

    monkeypatch.setattr(llm, "_structured_gemini", fake)
    st = run(synthetic_md, PRESETS["balanced"].weights, mode="ask", event_text="Iran strikes")
    assert st["review"] is None and st["brief_engine"] == "template"
    assert "손실은 약" not in st["brief"]


def test_llm_responses_are_cached(monkeypatch, tmp_path):
    monkeypatch.delenv("SENTINEL_DISABLE_LLM", raising=False)
    monkeypatch.setenv("GEMINI_API_KEY", "fake")
    monkeypatch.setattr("sentinel.config.LLM_CACHE", True)
    monkeypatch.setattr("sentinel.config.LLM_CACHE_DIR", tmp_path)
    monkeypatch.setattr(llm, "_CACHE", None)
    calls = []

    def fake(schema, system, user, smart):
        calls.append(1)
        return llm.plan_rules("Iran strikes", 0)

    monkeypatch.setattr(llm, "_structured_gemini", fake)
    a, _ = llm.structured_call(llm.Plan, "sys", "same prompt")
    b, _ = llm.structured_call(llm.Plan, "sys", "same prompt")
    c, _ = llm.structured_call(llm.Plan, "sys", "other prompt")
    assert len(calls) == 2 and a == b and c is not None
    monkeypatch.setattr(llm, "_CACHE", None)
