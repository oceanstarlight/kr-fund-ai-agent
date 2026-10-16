"""LangGraph agent: plan -> execute tools -> verify -> review.

    orchestrator -> sentinel (code) -> planner (LLM) -> executor (tools, code)
      -> verifier (rule checks, code) --fail--> planner (re-plan with feedback, max 2)
      -> reviewer (LLM: verdict + interpretation) -> brief (code) -> evaluate (replay only)

Every tool returns value + evidence + known unknowns (sentinel/tools.py). LLMs never
compute numbers; with no API key the planner and reviewer fall back to deterministic code.

Modes
- ``replay``: a library event is replayed as of its date. Only data visible before the
  event's base close is used for the forecast; realized outcomes are attached at the end.
- ``ask``: free-text event described by the user, forecast from the latest close.
- ``monitor``: scheduled check of the latest session; stops early when nothing triggers.
"""

from __future__ import annotations

import operator
import time
from concurrent.futures import ThreadPoolExecutor
from typing import Annotated, Any, TypedDict

import pandas as pd
from langgraph.graph import END, START, StateGraph

from sentinel import config, tools
from sentinel.agents import llm
from sentinel.data.store import MarketData
from sentinel.data.universe import ASSETS, FACTORS, normalize
from sentinel.events.library import Event, load_events
from sentinel.quant.engines import HorizonImpact, realized


class State(TypedDict, total=False):
    # inputs
    mode: str
    md: MarketData
    weights: dict[str, float]
    aum: float
    currency: str
    vol_scale: bool
    vol_method: str
    interval_method: str
    use_curated_tags: bool  # replay: False lets the planner classify the event text itself
    event_id: str
    event_text: str
    horizons: tuple[int, ...]
    # working state
    event: Event | None
    session: pd.Timestamp
    base: pd.Timestamp
    detection: dict
    plan: dict
    classification: dict
    classifier: str
    attempt: int
    analogs: list
    scenario: dict
    impacts: dict[int, HorizonImpact]
    betas: Any
    risk: dict
    tool_results: dict[str, tools.ToolResult]
    candidates: list[dict]
    verification: dict
    review: dict | None
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
    weights, dropped = md.usable_weights(normalize(state["weights"]))
    if not weights:
        raise ValueError("포트폴리오 자산 중 가격 데이터가 있는 자산이 없음")
    horizons = tuple(state.get("horizons") or config.HORIZONS)
    out: dict = {"weights": weights, "horizons": horizons, "attempt": 0, "mode": mode, "tool_results": {}}
    if mode == "replay":
        ev = next(e for e in load_events() if e.id == state["event_id"])
        session = md.trading_day_on_or_after(ev.t0)
        base = md.trading_day_before(ev.t0)
        out.update(event=ev, session=session, base=base, event_text=ev.description)
        msg = f"Replay: '{ev.name}' 기준일 {base.date()} 종가 이전 데이터만 사용"
    else:
        base = md.last_date
        out.update(event=None, session=base, base=base)
        msg = f"{'질의' if mode == 'ask' else '감시'} 모드: 최신 종가 {base.date()} 기준"
    if dropped:
        msg += f" / 가격 데이터 없음으로 제외 후 비중 재조정: {', '.join(dropped)}"
    return {**out, **_step("orchestrator", msg)}


def sentinel_node(state: State) -> dict:
    ev = state.get("event")
    r = tools.detect_signals(state["md"], state["session"], ev.date if ev else None)
    det = r.value
    top = ", ".join(f"{s['name']} z={s['z']:+.1f}" for s in det["top"])
    status = "이상 신호 감지" if det["triggered"] else "임계값 미달"
    return {"detection": det, "tool_results": {**state["tool_results"], r.tool: r},
            **_step("sentinel", f"{status}: {top} (코드, LLM 미사용)")}


def route_after_orchestrator(state: State) -> str:
    return "planner" if state["mode"] == "ask" else "sentinel"


def route_after_sentinel(state: State) -> str:
    if state["mode"] == "monitor" and not state["detection"]["triggered"]:
        return "end"
    return "planner"


def _signals_text(det: dict) -> str:
    parts = [f"{s['name']} 변동 z-score {s['z']:+.1f}" for s in det.get("top", [])]
    return "Market anomaly detected: " + "; ".join(parts)


def planner_node(state: State) -> dict:
    attempt = state.get("attempt", 0)
    ev: Event | None = state.get("event")
    use_tags = ev is not None and state.get("use_curated_tags", True)
    tags = {**ev.tags(), "headline_ko": ev.name, "rationale_ko": ev.note or ev.description} if use_tags else None
    text = state.get("event_text") or _signals_text(state.get("detection", {}))
    feedback = (state.get("verification") or {}).get("feedback") if attempt else None
    plan, engine = llm.plan_event(text, attempt, tags, feedback)
    cls = plan.classification.model_dump()
    cands = "; ".join(
        f"{c.label_ko}(최대 {c.analog_k}건, 유사도 {c.min_similarity:.2f}{', 동일 유형' if c.same_type_only else ''})"
        for c in plan.candidates
    )
    summary = (
        f"[계획 {attempt + 1}] 분류 {cls['type']}/{cls['region']}/{','.join(cls['channels'])} ({engine}); "
        f"검색 후보 {len(plan.candidates)}개: {cands} - {plan.rationale_ko}"
    )
    return {"plan": plan.model_dump(), "classification": cls, "classifier": engine, **_step("planner", summary)}


def rule_checks(impacts: dict, analogs: list, betas, weights: dict[str, float]) -> dict:
    """Deterministic checks shared by candidate selection and the verifier."""
    p = impacts.get(config.PRIMARY_HORIZON) or next(iter(impacts.values()))
    w = pd.Series(weights)
    checks = []
    n_ok = p.n_analogs >= config.ANALOG_MIN
    checks.append({"check": "유사 사건 수", "ok": n_ok, "detail": f"{p.n_analogs}건 (최소 {config.ANALOG_MIN})"})
    tol = max(config.ENGINE_GAP_ABS, config.ENGINE_GAP_REL * abs(0.5 * (p.engine_a["q50"] + p.engine_b["q50"])))
    gap_ok = bool(p.n_analogs) and p.engine_gap <= tol
    checks.append({"check": "엔진 A/B 교차검증", "ok": gap_ok, "detail": f"차이 {p.engine_gap:.2%}p (허용 {tol:.2%}p)"})
    beta_cov = float(w[betas.beta.dropna(how='all').index.intersection(w.index)].sum())
    checks.append({"check": "민감도 추정 커버리지", "ok": beta_cov >= 0.95, "detail": f"비중 {beta_cov:.0%}"})
    proxy_ok = p.backfilled_weight <= 0.2
    checks.append({"check": "프록시 데이터 비중", "ok": proxy_ok, "detail": f"최대 {p.backfilled_weight:.0%}"})
    sims = [a.similarity for a in analogs]
    mean_sim = sum(sims) / len(sims) if sims else 0.0
    checks.append({"check": "유사도 평균", "ok": mean_sim >= 0.35, "detail": f"{mean_sim:.2f} (기준 0.35)"})
    if n_ok and gap_ok and beta_cov >= 0.95 and proxy_ok and mean_sim >= 0.35 and p.n_analogs >= config.ANALOG_K:
        grade = "A"
    elif n_ok and gap_ok:
        grade = "B"
    else:
        grade = "C"
    return {"checks": checks, "passed": n_ok and gap_ok, "grade": grade, "mean_sim": mean_sim, "n": p.n_analogs}


def executor_node(state: State) -> dict:
    """Run the tools. Search candidates from the plan are executed in parallel and the first
    one that passes the rule checks is kept (no extra LLM call for a fallback search)."""
    plan = state["plan"]
    ev = state.get("event")
    ctx = tools.ToolContext(state["md"], state["weights"], state["base"], ev.id if ev else None)
    results = dict(state["tool_results"])
    r_be = tools.estimate_sensitivities(ctx)
    r_rk = tools.historical_risk(ctx, state["horizons"])

    def run_candidate(c: dict):
        r_an = tools.search_analogs(ctx, state["classification"], c["analog_k"], c["min_similarity"], c["same_type_only"])
        r_sc = tools.build_scenario(ctx, r_an.value, state["horizons"])
        r_im = tools.compute_impact(ctx, r_sc.value, r_be.value, state.get("vol_scale", True),
                                    state.get("vol_method", config.VOL_METHOD),
                                    state.get("interval_method", config.INTERVAL_METHOD))
        return c, r_an, r_sc, r_im, rule_checks(r_im.value, r_an.value, r_be.value, state["weights"])

    cands = plan["candidates"]
    if len(cands) > 1:
        with ThreadPoolExecutor(max_workers=len(cands)) as ex:
            outs = list(ex.map(run_candidate, cands))
    else:
        outs = [run_candidate(cands[0])]
    passing = [o for o in outs if o[4]["passed"]]
    # Keep the preferred passing candidate; if none pass, the one closest to passing.
    chosen = passing[0] if passing else max(outs, key=lambda o: (o[4]["n"] >= config.ANALOG_MIN, o[4]["mean_sim"]))
    c, r_an, r_sc, r_im, chk = chosen
    for r in (r_an, r_sc, r_be, r_im, r_rk):
        results[r.tool] = r
    p = r_im.value.get(config.PRIMARY_HORIZON) or next(iter(r_im.value.values()))
    cand_txt = ", ".join(f"{o[0]['label_ko']} {'통과' if o[4]['passed'] else '미통과'}({o[4]['n']}건)" for o in outs)
    names = ", ".join(f"{a.event.name}({a.similarity:.2f})" for a in r_an.value)
    summary = (
        f"후보 {len(outs)}개 병렬 실행 [{cand_txt}] → '{c['label_ko']}' 채택: 유사 사건 [{names}] → "
        f"D+{p.horizon} 엔진A {p.engine_a['q50']:+.2%}, 엔진B {p.engine_b['q50']:+.2%}, 통합 {p.blend['q50']:+.2%} "
        f"(불리 {p.blend['q25']:+.2%}); 보고된 한계 {len(tools.unknowns_of(list(results.values())))}건"
    )
    cand_log = [{"label": o[0]["label_ko"], "params": o[0], "passed": o[4]["passed"], "n_analogs": o[4]["n"],
                 "mean_similarity": o[4]["mean_sim"]} for o in outs]
    return {
        "analogs": r_an.value, "scenario": r_sc.value, "betas": r_be.value, "impacts": r_im.value,
        "risk": r_rk.value, "tool_results": results, "candidates": cand_log, **_step("executor", summary),
    }


def verifier_node(state: State) -> dict:
    chk = rule_checks(state["impacts"], state["analogs"], state["betas"], state["weights"])
    checks, grade = chk["checks"], chk["grade"]
    attempt = state.get("attempt", 0)
    retry = not chk["passed"] and attempt < config.MAX_VERIFY_RETRIES
    feedback = [f"{c['check']} 실패: {c['detail']}" for c in checks if not c["ok"]]
    verdict = "재계획 요청" if retry else f"통과 (규칙 등급 {grade})"
    summary = f"{verdict}: " + "; ".join(f"{c['check']} {'OK' if c['ok'] else 'NG'}({c['detail']})" for c in checks)
    out = {"verification": {"checks": checks, "grade": grade, "rule_grade": grade, "retry": retry,
                            "attempts": attempt + 1, "feedback": feedback}}
    if retry:
        out["attempt"] = attempt + 1
    return {**out, **_step("verifier", summary)}


def route_after_verifier(state: State) -> str:
    return "planner" if state["verification"]["retry"] else "reviewer"


_DOWNGRADE = {"A": "B", "B": "C", "C": "C"}


def reviewer_node(state: State) -> dict:
    facts = build_facts(state)
    unknowns = tools.unknowns_of(list(state["tool_results"].values()))
    context = {
        "classification": {k: state["classification"].get(k) for k in ("type", "region", "channels")},
        # Names, pass/fail and digit-free unknowns only: numeric details would tempt the LLM to copy numbers.
        "rule_checks": [{"check": c["check"], "ok": c["ok"]} for c in state["verification"]["checks"]],
        "known_unknowns": [{"code": u.code, "message": u.message} for u in unknowns],
    }
    review, problems = llm.review_result(facts, context)
    verification = dict(state["verification"])
    if review is None:
        if problems == ["llm unavailable"]:
            why = "LLM 미사용 → 규칙 등급 유지"
        elif any("error" in p or "no result" in p for p in problems):
            why = f"LLM 호출 실패 → 규칙 등급 유지 ({'; '.join(problems)})"
        else:
            why = f"LLM 검토문 감사 실패 → 폐기, 규칙 등급 유지 ({'; '.join(problems)})"
        return {"review": None, "verification": verification, **_step("reviewer", why)}
    if review.verdict == "caution":
        verification["grade"] = _DOWNGRADE[verification["grade"]]
    why = (f"LLM({llm.provider()}) 검토: {'주의' if review.verdict == 'caution' else '확인'}"
           f" → 최종 등급 {verification['grade']}; 우려 {len(review.concerns_ko)}건; 수치 감사 통과")
    return {"review": review.model_dump(), "verification": verification, **_step("reviewer", why)}


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


def unknowns_section(state: State) -> str:
    unk = tools.unknowns_of(list(state["tool_results"].values()))
    if not unk:
        return "### 도구가 보고한 한계\n- 없음"
    return "### 도구가 보고한 한계\n" + "\n".join(
        f"- {u.message}" + (f" ({u.detail})" if u.detail else "") for u in unk
    )


def brief_node(state: State) -> dict:
    facts = build_facts(state)
    review = state.get("review")
    parts = [summary_section(facts)]
    if review:
        parts.append(review["interpretation_md"])
        if review["concerns_ko"]:
            parts.append("### 검토 의견\n" + "\n".join(f"- {c}" for c in review["concerns_ko"]))
    parts += [data_sections(facts, state), unknowns_section(state), CAVEATS]
    engine = "llm" if review else "template"
    return {"facts": facts, "brief": "\n\n".join(parts), "brief_engine": engine,
            **_step("brief", "브리프 조립 (표·수치는 코드, 해석은 " + ("LLM)" if review else "생략)"))}


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
    g.add_node("planner", planner_node)
    g.add_node("executor", executor_node)
    g.add_node("verifier", verifier_node)
    g.add_node("reviewer", reviewer_node)
    g.add_node("brief", brief_node)
    g.add_node("evaluate", evaluate_node)
    g.add_edge(START, "orchestrator")
    g.add_conditional_edges("orchestrator", route_after_orchestrator, {"sentinel": "sentinel", "planner": "planner"})
    g.add_conditional_edges("sentinel", route_after_sentinel, {"planner": "planner", "end": END})
    g.add_edge("planner", "executor")
    g.add_edge("executor", "verifier")
    g.add_conditional_edges("verifier", route_after_verifier, {"planner": "planner", "reviewer": "reviewer"})
    g.add_edge("reviewer", "brief")
    g.add_conditional_edges("brief", route_after_brief, {"evaluate": "evaluate", "end": END})
    g.add_edge("evaluate", END)
    return g.compile()


def run(md: MarketData, weights: dict[str, float], mode: str = "replay", **kwargs) -> State:
    graph = build_graph()
    init: State = {"md": md, "weights": weights, "mode": mode, "trace": [], **kwargs}
    return graph.invoke(init, {"recursion_limit": 40})
