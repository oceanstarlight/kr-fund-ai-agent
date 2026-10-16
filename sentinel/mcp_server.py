"""MCP server exposing the deterministic tools to any agent (Claude, Gemini, in-house).

Each tool returns ``{"value": ..., "evidence": [...], "unknowns": [...]}`` so the calling
agent always sees what a number is based on and where it stops being reliable.

Run (stdio):  python -m sentinel.mcp_server
Uses the local data cache (build it first with scripts/build_snapshot.py).
"""

from __future__ import annotations

from functools import lru_cache

import pandas as pd
from mcp.server.mcpserver import MCPServer

from sentinel import config, tools
from sentinel.agents.pipeline import build_facts, run
from sentinel.data.store import MarketData
from sentinel.data.universe import ASSETS, PRESETS, normalize
from sentinel.events.library import load_events

server = MCPServer(
    name="macro-sentinel",
    instructions=(
        "Geopolitical/macro shock tools for institutional portfolios. All figures are computed "
        "from historical data; read 'unknowns' before relying on a number."
    ),
)


@lru_cache(maxsize=1)
def _md() -> MarketData:
    return MarketData.load(offline=True, max_age_hours=None)


def _weights(preset: str | None, weights: dict[str, float] | None) -> dict[str, float]:
    if weights:
        return normalize(weights)
    return PRESETS[preset or "sovereign"].weights


def _as_of(as_of: str | None) -> pd.Timestamp:
    md = _md()
    return md.last_date if not as_of else (md.trading_day_before(pd.Timestamp(as_of) + pd.Timedelta(days=1)) or md.last_date)


def _pack(r: tools.ToolResult, value) -> dict:
    return {"value": value, "evidence": r.evidence, "unknowns": [u.as_dict() for u in r.unknowns]}


@server.tool(description="List the historical event library (id, date, type, region, channels).")
def list_events() -> list[dict]:
    return [{"id": e.id, "name": e.name, "t0": str(e.t0.date()), **e.tags()} for e in load_events()]


@server.tool(description="List supported assets and portfolio presets.")
def list_universe() -> dict:
    return {
        "assets": {t: {"name": a.name, "class": a.asset_class, "region": a.region} for t, a in ASSETS.items()},
        "presets": {k: {"name": p.name, "weights": p.weights} for k, p in PRESETS.items()},
    }


@server.tool(description="Anomaly detection on market factors and the daily geopolitical risk index for one session.")
def detect_signals(session: str | None = None) -> dict:
    d = _as_of(session)
    r = tools.detect_signals(_md(), d)
    return _pack(r, r.value)


@server.tool(description="Find historical analog events by tags, using only events observable before as_of.")
def search_analogs(event_type: str, region: str, channels: list[str], k: int = config.ANALOG_K,
                   min_similarity: float = config.ANALOG_MIN_SIMILARITY, as_of: str | None = None) -> dict:
    ctx = tools.ToolContext(_md(), PRESETS["sovereign"].weights, _as_of(as_of))
    r = tools.search_analogs(ctx, {"type": event_type, "region": region, "channels": channels}, k, min_similarity)
    return _pack(r, [{"id": a.event.id, "name": a.event.name, "similarity": a.similarity, "reasons": list(a.reasons)}
                     for a in r.value])


@server.tool(description="Estimate portfolio impact (engine A factor model, engine B analog replay) for an event "
                         "described by tags. Deterministic, no LLM.")
def portfolio_impact(event_type: str, region: str, channels: list[str], preset: str | None = "sovereign",
                     weights: dict[str, float] | None = None, as_of: str | None = None) -> dict:
    w = _weights(preset, weights)
    ctx = tools.ToolContext(_md(), w, _as_of(as_of))
    r_an = tools.search_analogs(ctx, {"type": event_type, "region": region, "channels": channels})
    r_sc = tools.build_scenario(ctx, r_an.value)
    r_be = tools.estimate_sensitivities(ctx)
    r_im = tools.compute_impact(ctx, r_sc.value, r_be.value)
    value = {
        f"D+{h}": {"blend": p.blend, "engine_a": p.engine_a, "engine_b": p.engine_b, "n_analogs": p.n_analogs,
                   "contribution": p.contribution.round(6).to_dict()}
        for h, p in r_im.value.items()
    }
    unknowns = [u.as_dict() for u in tools.unknowns_of([r_an, r_sc, r_be, r_im])]
    evidence = r_an.evidence + r_be.evidence + r_im.evidence
    return {"value": value, "evidence": evidence, "unknowns": unknowns}


@server.tool(description="Run the full agent (planner, tools, verifier, reviewer) on a free-text event. "
                         "Uses an LLM only if an API key is configured.")
def analyze_event(description: str, preset: str | None = "sovereign", weights: dict[str, float] | None = None) -> dict:
    st = run(_md(), _weights(preset, weights), mode="ask", event_text=description)
    return {
        "classification": st["classification"],
        "facts": build_facts(st),
        "verification": st["verification"],
        "unknowns": [u.as_dict() for u in tools.unknowns_of(list(st["tool_results"].values()))],
        "brief_markdown": st["brief"],
    }


if __name__ == "__main__":
    server.run("stdio")
