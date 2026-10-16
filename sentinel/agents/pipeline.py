"""LangGraph pipeline: orchestrator -> sentinel -> analyst -> scenario -> quant -> verifier
(-> analyst retry loop) -> brief -> evaluate (replay only).

Modes
- ``replay``: a library event is replayed as of its date. Only data visible before the
  event's base close is used for the forecast; realized outcomes are attached at the end.
- ``ask``: free-text event described by the user, forecast from the latest close.
- ``monitor``: scheduled check of the latest session; stops early when nothing triggers.
"""

from __future__ import annotations

import operator
import time
from typing import Annotated, Any, TypedDict

import pandas as pd
from langgraph.graph import END, START, StateGraph

from sentinel import config
from sentinel.agents import llm
from sentinel.data.store import MarketData
from sentinel.data.universe import ASSETS, FACTORS, normalize
from sentinel.events.library import Event, find_analogs, load_events
from sentinel.monitor import detect
from sentinel.quant import betas as betas_mod
from sentinel.quant.engines import HorizonImpact, portfolio_impact, realized
from sentinel.quant.risk import historical_var
from sentinel.quant.scenario import analog_eligibility, analog_windows, scenario_table


class State(TypedDict, total=False):
    # inputs
    mode: str
    md: MarketData
    weights: dict[str, float]
    aum: float
    currency: str
    vol_scale: bool
    event_id: str
    event_text: str
    horizons: tuple[int, ...]
    # working state
    event: Event | None
    session: pd.Timestamp
    base: pd.Timestamp
    detection: dict
    classification: dict
    classifier: str
    attempt: int
    analogs: list
    scenario: dict
    impacts: dict[int, HorizonImpact]
    betas: Any
    risk: dict
    verification: dict
    facts: dict
    brief: str
    brief_engine: str
    realized: dict
    trace: Annotated[list[dict], operator.add]


def _step(node: str, summary: str, **extra) -> dict:
    return {"trace": [{"node": node, "summary": summary, "time": time.strftime("%H:%M:%S"), **extra}]}


# ----------------------------------------------------------------------------- nodes
def orchestrator(state: State) -> dict:
    md = state["md"]
    mode = state.get("mode", "replay")
    weights = normalize(state["weights"])
    horizons = tuple(state.get("horizons") or config.HORIZONS)
    out: dict = {"weights": weights, "horizons": horizons, "attempt": 0, "mode": mode}
    if mode == "replay":
        ev = next(e for e in load_events() if e.id == state["event_id"])
        session = md.trading_day_on_or_after(ev.t0)
        base = md.trading_day_before(ev.t0)
        out.update(event=ev, session=session, base=base, event_text=ev.description)
        plan = f"Replay: '{ev.name}' 기준일 {base.date()} 종가 이전 데이터만 사용"
    else:
        base = md.last_date
        out.update(event=None, session=base, base=base)
        plan = f"{'질의' if mode == 'ask' else '감시'} 모드: 최신 종가 {base.date()} 기준"
    return {**out, **_step("orchestrator", plan)}


def sentinel_node(state: State) -> dict:
    md = state["md"]
    ev = state.get("event")
    session = state["session"]
    det = detect(md.truncate(session), session, ev.date if ev else None)
    top = ", ".join(f"{s['name']} z={s['z']:+.1f}" for s in det["top"])
    status = "이상 신호 감지" if det["triggered"] else "임계값 미달"
    return {"detection": det, **_step("sentinel", f"{status}: {top}")}


def route_after_sentinel(state: State) -> str:
    if state["mode"] == "monitor" and not state["detection"]["triggered"]:
        return "end"
    return "analyst"


def route_after_orchestrator(state: State) -> str:
    return "analyst" if state["mode"] == "ask" else "sentinel"


def _signals_text(det: dict) -> str:
    parts = [f"{s['name']} 변동 z-score {s['z']:+.1f}" for s in det.get("top", [])]
    return "Market anomaly detected: " + "; ".join(parts)


def analyst_node(state: State) -> dict:
    md = state["md"]
    attempt = state.get("attempt", 0)
    ev: Event | None = state.get("event")
    if "classification" in state and state.get("classification"):
        cls, engine = state["classification"], state["classifier"]
    elif ev is not None:
        cls, engine = {**ev.tags(), "headline_ko": ev.name, "rationale_ko": ev.note or ev.description}, "library"
    else:
        text = state.get("event_text") or _signals_text(state.get("detection", {}))
        c, engine = llm.classify_event(text)
        cls = c.model_dump()

    # Retry strategy chosen by the verifier loop: widen or tighten the analog set.
    k, min_sim, same_type = config.ANALOG_K, config.ANALOG_MIN_SIMILARITY, False
    if attempt == 1:
        k, min_sim, same_type = config.ANALOG_K + 2, 0.15, True
    elif attempt >= 2:
        k, min_sim, same_type = config.ANALOG_K + 5, 0.05, False

    eligible = analog_eligibility(md, state["base"], max(state["horizons"]))
    analogs = find_analogs(
        cls, load_events(), cutoff=state["base"], k=k, min_similarity=min_sim,
        exclude_id=ev.id if ev else None, same_type_only=same_type, eligible=eligible,
    )
    if attempt == 1 and len(analogs) < config.ANALOG_MIN:
        analogs = find_analogs(
            cls, load_events(), cutoff=state["base"], k=k, min_similarity=min_sim,
            exclude_id=ev.id if ev else None, eligible=eligible,
        )
    names = ", ".join(f"{a.event.name}({a.similarity:.2f})" for a in analogs)
    summary = f"[시도 {attempt + 1}] 분류 {cls['type']}/{cls['region']} ({engine}) → 유사 사건 {len(analogs)}건: {names}"
    return {"classification": cls, "classifier": engine, "analogs": analogs, **_step("analyst", summary)}


def scenario_node(state: State) -> dict:
    md = state["md"].truncate(state["base"])
    scen = {}
    for h in state["horizons"]:
        scen[h] = {"windows": analog_windows(md, state["analogs"], h)}
        scen[h]["table"] = scenario_table(scen[h]["windows"])
    t = scen[config.PRIMARY_HORIZON]["table"] if config.PRIMARY_HORIZON in scen else next(iter(scen.values()))["table"]
    if t.empty:
        summary = "유사 사건 구간 없음"
    else:
        summary = "D+%d 기본 시나리오(중앙값): " % config.PRIMARY_HORIZON + ", ".join(
            f"{r['name']} {_fmt_factor(r['q50'], k)}" for k, r in t.iterrows()
        )
    return {"scenario": scen, **_step("scenario_builder", summary)}


def quant_node(state: State) -> dict:
    md = state["md"].truncate(state["base"])
    w = state["weights"]
    b = betas_mod.estimate_betas(md, state["base"], list(w))
    impacts, risk = {}, {}
    for h in state["horizons"]:
        impacts[h] = portfolio_impact(md, w, state["scenario"][h]["windows"], b, h, state.get("vol_scale", True))
        risk[h] = historical_var(md, w, state["base"], h)
    p = impacts.get(config.PRIMARY_HORIZON) or next(iter(impacts.values()))
    summary = (
        f"D+{p.horizon} 포트폴리오: 엔진A {p.engine_a['q50']:+.2%}, 엔진B {p.engine_b['q50']:+.2%}, "
        f"통합 {p.blend['q50']:+.2%} (불리 {p.blend['q25']:+.2%})"
    )
    return {"betas": b, "impacts": impacts, "risk": risk, **_step("quant_engine", summary)}


def verifier_node(state: State) -> dict:
    p = state["impacts"].get(config.PRIMARY_HORIZON) or next(iter(state["impacts"].values()))
    b = state["betas"]
    w = pd.Series(state["weights"])
    checks = []
    n_ok = p.n_analogs >= config.ANALOG_MIN
    checks.append({"check": "유사 사건 수", "ok": n_ok, "detail": f"{p.n_analogs}건 (최소 {config.ANALOG_MIN})"})
    tol = max(config.ENGINE_GAP_ABS, config.ENGINE_GAP_REL * abs(0.5 * (p.engine_a["q50"] + p.engine_b["q50"])))
    gap_ok = bool(p.n_analogs) and p.engine_gap <= tol
    checks.append({"check": "엔진 A/B 교차검증", "ok": gap_ok, "detail": f"차이 {p.engine_gap:.2%}p (허용 {tol:.2%}p)"})
    beta_cov = float(w[b.beta.dropna(how='all').index.intersection(w.index)].sum())
    checks.append({"check": "민감도 추정 커버리지", "ok": beta_cov >= 0.95, "detail": f"비중 {beta_cov:.0%}"})
    proxy_ok = p.backfilled_weight <= 0.2
    checks.append({"check": "프록시 데이터 비중", "ok": proxy_ok, "detail": f"최대 {p.backfilled_weight:.0%}"})

    sims = [a.similarity for a in state["analogs"]]
    mean_sim = sum(sims) / len(sims) if sims else 0.0
    checks.append({"check": "유사도 평균", "ok": mean_sim >= 0.35, "detail": f"{mean_sim:.2f} (기준 0.35)"})

    attempt = state.get("attempt", 0)
    retry = (not n_ok or not gap_ok) and attempt < config.MAX_VERIFY_RETRIES
    if n_ok and gap_ok and beta_cov >= 0.95 and proxy_ok and mean_sim >= 0.35 and p.n_analogs >= config.ANALOG_K:
        grade = "A"
    elif n_ok and gap_ok:
        grade = "B"
    else:
        grade = "C"
    verdict = "재분석 요청" if retry else f"통과 (신뢰도 {grade})"
    summary = f"{verdict}: " + "; ".join(f"{c['check']} {'OK' if c['ok'] else 'NG'}({c['detail']})" for c in checks)
    out = {"verification": {"checks": checks, "grade": grade, "retry": retry, "attempts": attempt + 1}}
    if retry:
        out["attempt"] = attempt + 1
    return {**out, **_step("verifier", summary)}


def route_after_verifier(state: State) -> str:
    return "analyst" if state["verification"]["retry"] else "brief"


def _fmt_factor(v: float, key: str) -> str:
    f = FACTORS[key]
    if f.transform == "logret":
        return f"{v:+.1%}"
    return f"{v:+.2f}{f.unit}"


def format_money(x: float, currency: str = "USD") -> str:
    if currency == "KRW":
        for unit, div in (("조원", 1e12), ("억원", 1e8)):
            if abs(x) >= div:
                return f"{x / div:+,.1f}{unit}"
        return f"{x:+,.0f}원"
    # No "$" sign: Streamlit markdown would treat paired dollar signs as LaTeX.
    for unit, div in (("B", 1e9), ("M", 1e6)):
        if abs(x) >= div:
            return f"USD {x / div:+,.2f}{unit}"
    return f"USD {x:+,.0f}"


def build_facts(state: State) -> dict[str, str]:
    aum = state.get("aum") or 0.0
    cur = state.get("currency", "USD")
    cls = state["classification"]
    facts: dict[str, str] = {
        "event_name": state["event"].name if state.get("event") else cls.get("headline_ko", ""),
        "base_date": str(state["base"].date()),
        "n_analogs": str(state["impacts"][next(iter(state["impacts"]))].n_analogs),
        "grade": state["verification"]["grade"],
        "analog_names": ", ".join(a.event.name for a in state["analogs"]),
    }
    if state.get("event"):
        facts["event_date"] = str(state["event"].date.date())
    for h, p in state["impacts"].items():
        facts[f"port_q50_{h}d"] = f"{p.blend['q50']:+.2%}"
        facts[f"port_q25_{h}d"] = f"{p.blend['q25']:+.2%}"
        facts[f"port_q75_{h}d"] = f"{p.blend['q75']:+.2%}"
        facts[f"engine_a_{h}d"] = f"{p.engine_a['q50']:+.2%}"
        facts[f"engine_b_{h}d"] = f"{p.engine_b['q50']:+.2%}"
        if aum:
            facts[f"pnl_q50_{h}d"] = format_money(p.blend["q50"] * aum, cur)
            facts[f"pnl_q25_{h}d"] = format_money(p.blend["q25"] * aum, cur)
        r = state["risk"].get(h, {})
        if r.get("var") == r.get("var"):
            facts[f"var95_{h}d"] = f"{-r['var']:+.2%}"
    p = state["impacts"].get(config.PRIMARY_HORIZON) or next(iter(state["impacts"].values()))
    losses = p.contribution[p.contribution < 0]
    for i, (t, c) in enumerate(losses.head(3).items(), 1):
        facts[f"top_loss_{i}_asset"] = ASSETS[t].name
        facts[f"top_loss_{i}_contrib"] = f"{c:+.2%}"
    table = state["scenario"].get(config.PRIMARY_HORIZON, {}).get("table")
    if table is not None and not table.empty:
        for k, r in table.iterrows():
            facts[f"factor_{k}_q50"] = _fmt_factor(r["q50"], k)
    return facts


def summary_section(facts: dict[str, str]) -> str:
    h = config.PRIMARY_HORIZON
    return "\n".join([
        "### 요약",
        f"- 사건: **{facts['event_name']}**" + (f" ({facts['event_date']})" if "event_date" in facts else ""),
        f"- 예측 기준: {facts['base_date']} 종가까지의 데이터, 유사 과거 사건 {facts['n_analogs']}건",
        f"- D+{h} 포트폴리오 예상 영향: **{facts[f'port_q50_{h}d']}** (불리 시나리오 {facts[f'port_q25_{h}d']})",
    ])


def data_sections(facts: dict[str, str], state: State) -> str:
    """Numeric tables are always rendered by code, never by the LLM."""
    h = config.PRIMARY_HORIZON
    lines = [
        "### 예상 영향",
        "| 기간 | 기본(중앙값) | 불리(하위 25%) | 유리(상위 25%) | 엔진A | 엔진B | 과거 VaR95 |",
        "|---|---|---|---|---|---|---|",
    ]
    for hh in state["horizons"]:
        lines.append(
            f"| D+{hh} | {facts[f'port_q50_{hh}d']} | {facts[f'port_q25_{hh}d']} | {facts[f'port_q75_{hh}d']} "
            f"| {facts[f'engine_a_{hh}d']} | {facts[f'engine_b_{hh}d']} | {facts.get(f'var95_{hh}d', '-')} |"
        )
    if f"pnl_q50_{h}d" in facts:
        lines.append(f"\n- 운용자산 기준 D+{h} 예상 손익: {facts[f'pnl_q50_{h}d']} (불리 {facts[f'pnl_q25_{h}d']})")
    lines += ["", "### 주요 기여 자산 (손실 기여 상위)"]
    if "top_loss_1_asset" not in facts:
        lines.append("- 손실 기여 자산 없음")
    for i in (1, 2, 3):
        if f"top_loss_{i}_asset" in facts:
            lines.append(f"{i}. {facts[f'top_loss_{i}_asset']}: {facts[f'top_loss_{i}_contrib']}")
    lines += [
        "",
        "### 근거와 신뢰도",
        f"- 유사 사건: {facts['analog_names']}",
        f"- 검증 등급: **{facts['grade']}** (엔진 A 팩터모델 / 엔진 B 과거사건 재현 교차검증)",
    ]
    return "\n".join(lines)


CAVEATS = "\n".join([
    "### 유의사항",
    "- 통계적 연관성에 기반한 추정이며 인과를 보장하지 않음. 투자 판단 보조용.",
    "- 대체자산은 상장 프록시로 평가함.",
])


def template_brief(facts: dict[str, str], state: State) -> str:
    return "\n\n".join([summary_section(facts), data_sections(facts, state), CAVEATS])


def brief_node(state: State) -> dict:
    facts = build_facts(state)
    context = {
        "classification": {k: state["classification"].get(k) for k in ("type", "region", "channels")},
        # Only names and pass/fail: numeric details would tempt the LLM to copy numbers.
        "verification": [{"check": c["check"], "ok": c["ok"]} for c in state["verification"]["checks"]],
    }
    narrative, problems = llm.write_brief_llm(facts, context)
    if narrative is None:
        text, engine = template_brief(facts, state), "template"
        if problems == ["llm unavailable"]:
            why = "LLM 미사용"
        elif any("error" in p or "no text" in p for p in problems):
            why = f"LLM 호출 실패 → 템플릿 대체 ({'; '.join(problems)})"
        else:
            why = f"LLM 결과 감사 실패 → 템플릿 대체 ({'; '.join(problems)})"
    else:
        text = "\n\n".join([summary_section(facts), narrative, data_sections(facts, state), CAVEATS])
        engine, why = "llm", f"LLM({llm.provider()}) 해설 작성, 수치 감사 통과"
    return {"facts": facts, "brief": text, "brief_engine": engine, **_step("brief", why)}


def route_after_brief(state: State) -> str:
    return "evaluate" if state["mode"] == "replay" else "end"


def evaluate_node(state: State) -> dict:
    md = state["md"]
    ev = state["event"]
    out = {}
    for h in state["horizons"]:
        r = realized(md, state["weights"], ev.t0, h)
        if r is None:
            continue
        p = state["impacts"][h]
        out[h] = {
            "realized": r["portfolio"],
            "predicted": p.blend["q50"],
            "error": p.blend["q50"] - r["portfolio"],
            "in_iqr": p.blend["q25"] <= r["portfolio"] <= p.blend["q75"],
            "direction_hit": (p.blend["q50"] >= 0) == (r["portfolio"] >= 0),
            "assets": r["assets"],
            "factors": r["factors"],
        }
    summary = "; ".join(
        f"D+{h} 예측 {v['predicted']:+.2%} / 실측 {v['realized']:+.2%}" for h, v in out.items()
    ) or "실측 데이터 없음"
    return {"realized": out, **_step("evaluate", summary)}


# ----------------------------------------------------------------------------- graph
def build_graph():
    g = StateGraph(State)
    g.add_node("orchestrator", orchestrator)
    g.add_node("sentinel", sentinel_node)
    g.add_node("analyst", analyst_node)
    g.add_node("scenario_builder", scenario_node)
    g.add_node("quant_engine", quant_node)
    g.add_node("verifier", verifier_node)
    g.add_node("brief", brief_node)
    g.add_node("evaluate", evaluate_node)
    g.add_edge(START, "orchestrator")
    g.add_conditional_edges("orchestrator", route_after_orchestrator, {"sentinel": "sentinel", "analyst": "analyst"})
    g.add_conditional_edges("sentinel", route_after_sentinel, {"analyst": "analyst", "end": END})
    g.add_edge("analyst", "scenario_builder")
    g.add_edge("scenario_builder", "quant_engine")
    g.add_edge("quant_engine", "verifier")
    g.add_conditional_edges("verifier", route_after_verifier, {"analyst": "analyst", "brief": "brief"})
    g.add_conditional_edges("brief", route_after_brief, {"evaluate": "evaluate", "end": END})
    g.add_edge("evaluate", END)
    return g.compile()


def run(md: MarketData, weights: dict[str, float], mode: str = "replay", **kwargs) -> State:
    graph = build_graph()
    init: State = {"md": md, "weights": weights, "mode": mode, "trace": [], **kwargs}
    return graph.invoke(init, {"recursion_limit": 40})
