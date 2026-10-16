"""Deterministic tool layer.

Every tool returns a ``ToolResult`` carrying the value plus the evidence it is based on and
the known unknowns (limits of the claim). The planner and the final reviewer reason over
evidence and unknowns; numbers are only produced here. The same functions are exposed to
external agents through the MCP server in ``sentinel/mcp_server.py``.

Unknown messages are written without digits on purpose: they are passed to the LLM
reviewer, whose output must not contain free-typed numbers. Numeric details go to
``detail`` for the UI.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd

from sentinel import config
from sentinel.data.store import MarketData
from sentinel.events.library import find_analogs, load_events
from sentinel.monitor import detect
from sentinel.quant import betas as betas_mod
from sentinel.quant.engines import HorizonImpact, portfolio_impact
from sentinel.quant.risk import historical_var
from sentinel.quant.scenario import analog_eligibility, analog_windows, scenario_table


@dataclass
class Unknown:
    code: str
    message: str  # digit-free, safe to show the LLM
    detail: str = ""  # may contain numbers, UI only

    def as_dict(self) -> dict:
        return {"code": self.code, "message": self.message, "detail": self.detail}


@dataclass
class ToolResult:
    tool: str
    value: Any
    evidence: list[str] = field(default_factory=list)
    unknowns: list[Unknown] = field(default_factory=list)

    def summary(self) -> dict:
        """Digit-free view for the LLM reviewer."""
        return {"tool": self.tool, "unknowns": [{"code": u.code, "message": u.message} for u in self.unknowns]}


@dataclass
class ToolContext:
    md: MarketData  # full data; tools truncate to ``base`` themselves
    weights: dict[str, float]
    base: pd.Timestamp  # forecast base close; nothing after it may be used
    exclude_event_id: str | None = None

    @property
    def visible(self) -> MarketData:
        return self.md.truncate(self.base)


TOOL_CATALOG = {
    "detect_signals": "시장 변수·지정학 리스크 지수의 이상치(z-score) 감지",
    "search_analogs": "유형·지역·전이경로 태그로 과거 유사 사건 검색 (예측 기준일 이전에 관측이 끝난 사건만)",
    "build_scenario": "유사 사건 구간의 팩터 실측 변화로 시나리오 분포 생성",
    "estimate_sensitivities": "자산별 팩터 민감도(주간 Ridge 회귀) 추정",
    "compute_impact": "엔진A(팩터모델)·엔진B(사건재현) 포트폴리오 영향과 기여도",
    "historical_risk": "과거 수익률 기반 VaR/CVaR",
}


# ------------------------------------------------------------------------------ tools
def detect_signals(md: MarketData, session: pd.Timestamp, event_date: pd.Timestamp | None = None) -> ToolResult:
    det = detect(md.truncate(session), session, event_date)
    ev = [f"시장 데이터 {session.date()} 종가", f"팩터 {len(md.factors.columns)}종(EIA·연준·Cboe·ETF 파생), 감시 자산 4종"]
    unk = []
    if md.gpr is None:
        unk.append(Unknown("gpr_missing", "지정학 리스크 지수 데이터가 없어 시장 신호만 사용"))
    else:
        ev.append(f"GPR 일간 지수 (최신 {md.gpr.index[-1].date()})")
        if md.gpr.index[-1] < pd.Timestamp(session) - pd.Timedelta(days=3):
            unk.append(Unknown("gpr_stale", "지정학 리스크 지수 갱신이 늦어 최근 며칠은 반영되지 않음",
                               f"최신 {md.gpr.index[-1].date()}"))
    return ToolResult("detect_signals", det, ev, unk)


def search_analogs(
    ctx: ToolContext,
    query: dict,
    k: int = config.ANALOG_K,
    min_similarity: float = config.ANALOG_MIN_SIMILARITY,
    same_type_only: bool = False,
) -> ToolResult:
    k = int(np.clip(k, config.ANALOG_MIN, 10))
    min_similarity = float(np.clip(min_similarity, 0.0, 0.8))
    eligible = analog_eligibility(ctx.md, ctx.base, max(config.HORIZONS))
    events = load_events()
    analogs = find_analogs(query, events, ctx.base, k, min_similarity, ctx.exclude_event_id, same_type_only, eligible)
    unk = []
    if len(analogs) < config.ANALOG_MIN:
        unk.append(Unknown("few_analogs", "조건에 맞는 과거 유사 사건이 부족함", f"{len(analogs)}건"))
    sims = [a.similarity for a in analogs]
    if sims and float(np.mean(sims)) < 0.35:
        unk.append(Unknown("low_similarity", "선정된 유사 사건과 현재 사건의 태그 유사도가 낮음", f"평균 {np.mean(sims):.2f}"))
    if analogs and not any(a.event.type == query.get("type") for a in analogs):
        unk.append(Unknown("no_same_type", "같은 유형의 과거 사건이 없어 다른 유형으로 대체함"))
    ev = [f"사건 라이브러리 {len(events)}건 중 기준일 이전 관측 종료 사건만 사용"] + [
        f"{a.event.id} (유사도 {a.similarity:.2f}: {'; '.join(a.reasons)})" for a in analogs
    ]
    return ToolResult("search_analogs", analogs, ev, unk)


def build_scenario(ctx: ToolContext, analogs: list, horizons=config.HORIZONS) -> ToolResult:
    md = ctx.visible
    out, unk = {}, []
    for h in horizons:
        windows = analog_windows(md, analogs, h)
        out[h] = {"windows": windows, "table": scenario_table(windows)}
        if any(w.factor_change.isna().any() for w in windows):
            unk.append(Unknown("factor_gap", "일부 유사 사건 구간에 팩터 데이터가 비어 있어 해당 팩터 변화는 영으로 처리",
                               f"D+{h}"))
    ev = [f"D+{h}: {len(v['windows'])}개 구간" for h, v in out.items()]
    return ToolResult("build_scenario", out, ev, unk)


def _quality_unknowns(ctx: ToolContext) -> list[Unknown]:
    """Data-quality failures on the portfolio's assets or the factors, visible at ``base``."""
    rel = [f for f in ctx.md.quality
           if (f.source != "yahoo" or f.key in ctx.weights) and f.relevant_at(ctx.base)]
    out = []
    for sev, code, msg in [("error", "data_quality_error", "데이터 품질 검사에서 오류가 난 시계열이 있어 해당 값의 신뢰도가 낮음"),
                           ("warning", "data_quality_warning", "데이터 품질 검사에서 이상치 경고가 난 시계열이 있음")]:
        hits = [f for f in rel if f.severity == sev]
        if hits:
            out.append(Unknown(code, msg, "; ".join(f"{f.key}:{f.check}" for f in hits)[:200]))
    return out


def estimate_sensitivities(ctx: ToolContext) -> ToolResult:
    b = betas_mod.estimate_betas(ctx.visible, ctx.base, list(ctx.weights))
    w = pd.Series(ctx.weights)
    unk = _quality_unknowns(ctx)
    low = b.r2[b.r2 < 0.1].index.intersection(w.index)
    if float(w[low].sum()) > 0.1:
        unk.append(Unknown("low_r2", "팩터모델 설명력이 낮은 자산의 비중이 큼", f"{', '.join(low)} (비중 {w[low].sum():.0%})"))
    short = b.n_obs[b.n_obs < config.BETA_MIN_WEEKS].index.intersection(w.index)
    if len(short):
        unk.append(Unknown("short_history", "추정 기간 데이터가 부족한 자산이 있음", ", ".join(short)))
    bf = b.backfilled_share.reindex(w.index).fillna(0)
    if float((w * bf).sum()) > 0.05:
        unk.append(Unknown("proxy_in_betas", "민감도 추정 일부가 상장 이전 프록시 데이터에 기반함", f"{(w * bf).sum():.0%}"))
    if config.RIDGE_ALPHA == "cv":
        a = b.alpha.reindex(w.index).dropna()
        reg = f"Ridge alpha 자산별 교차검증(LOO) 선택, 중앙값 {a.median():g}" if len(a) else "Ridge alpha 교차검증"
    else:
        reg = f"Ridge alpha {config.RIDGE_ALPHA}"
    ev = [f"주간 수익률 최근 {config.BETA_WINDOW_WEEKS}주 (기준일 {ctx.base.date()} 이전), {reg}"]
    return ToolResult("estimate_sensitivities", b, ev, unk)


def compute_impact(
    ctx: ToolContext, scenario: dict, betas, use_vol_scale: bool = True,
    vol_method: str = config.VOL_METHOD, interval_method: str = config.INTERVAL_METHOD,
) -> ToolResult:
    md = ctx.visible
    impacts: dict[int, HorizonImpact] = {}
    unk = []
    for h, sc in scenario.items():
        impacts[h] = portfolio_impact(md, ctx.weights, sc["windows"], betas, h, use_vol_scale, vol_method, interval_method)
    p = impacts.get(config.PRIMARY_HORIZON) or next(iter(impacts.values()))
    if p.notes:
        unk.append(Unknown("engine_b_fallback", "일부 자산은 과거 사건 당시 데이터가 없어 엔진A 값으로 대체", "; ".join(p.notes)[:200]))
    if p.backfilled_weight > 0.2:
        unk.append(Unknown("proxy_heavy", "유사 사건 당시 데이터 중 프록시로 보완한 비중이 큼", f"{p.backfilled_weight:.0%}"))
    if p.n_analogs and p.engine_gap > max(config.ENGINE_GAP_ABS, config.ENGINE_GAP_REL * abs(p.blend["q50"])):
        unk.append(Unknown("engine_gap", "두 엔진의 추정치 차이가 허용 범위를 넘음", f"{p.engine_gap:.2%}p"))
    if 20 in impacts:
        unk.append(Unknown("long_horizon", "D+20 추정은 백테스트상 단순 기준선보다 오차가 커서 참고용"))
    if not p.per_analog.empty and (p.per_analog["vol_scale"] >= 1.99).all():
        unk.append(Unknown("vol_regime", "현재 변동성이 과거 유사 사건 당시보다 높아 보정 배수가 상한에 걸림"))
    ev = ["엔진A: 민감도 × 유사 사건 팩터 변화", "엔진B: 유사 사건 당시 자산 실측 수익률",
          "변동성 국면 보정 " + (f"적용({vol_method})" if use_vol_scale else "미적용"),
          f"예측 구간: {'유사 사건 분산 + GARCH 변동성 예측' if p.interval_method == 'garch' else '유사 사건 분위수'}"]
    return ToolResult("compute_impact", impacts, ev, unk)


def historical_risk(ctx: ToolContext, horizons=config.HORIZONS) -> ToolResult:
    risk = {h: historical_var(ctx.visible, ctx.weights, ctx.base, h) for h in horizons}
    return ToolResult("historical_risk", risk, ["최근 3년 중첩 구간 과거 시뮬레이션, 신뢰수준 95%"], [])


def unknowns_of(results: list[ToolResult]) -> list[Unknown]:
    seen, out = set(), []
    for r in results:
        for u in r.unknowns:
            if u.code not in seen:
                seen.add(u.code)
                out.append(u)
    return out
