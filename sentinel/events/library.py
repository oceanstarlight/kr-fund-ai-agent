"""Historical event library and analog search."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Callable
from pathlib import Path

import pandas as pd

LIBRARY_PATH = Path(__file__).with_name("events.json")


@dataclass(frozen=True)
class Event:
    id: str
    name: str
    date: pd.Timestamp
    t0: pd.Timestamp
    type: str
    region: str
    channels: frozenset[str]
    description: str
    lat: float | None = None
    lon: float | None = None
    note: str = ""
    # Why the event is in the library: "curated" (initial set), "gpr_spike" (scripts/gpr_candidates.py
    # rule), "official" (dated official announcement in an under-represented type).
    selection: str = "curated"

    def tags(self) -> dict:
        return {"type": self.type, "region": self.region, "channels": sorted(self.channels)}


@dataclass(frozen=True)
class Analog:
    event: Event
    similarity: float
    weight: float = 0.0
    reasons: tuple[str, ...] = field(default_factory=tuple)


def load_schema() -> dict:
    return json.loads(LIBRARY_PATH.read_text(encoding="utf-8"))["schema"]


def load_events() -> list[Event]:
    raw = json.loads(LIBRARY_PATH.read_text(encoding="utf-8"))["events"]
    out = [
        Event(
            id=e["id"],
            name=e["name"],
            date=pd.Timestamp(e["date"]),
            t0=pd.Timestamp(e["t0"]),
            type=e["type"],
            region=e["region"],
            channels=frozenset(e["channels"]),
            description=e["description"],
            lat=e.get("lat"),
            lon=e.get("lon"),
            note=e.get("note", ""),
            selection=e.get("selection", "curated"),
        )
        for e in raw
    ]
    return sorted(out, key=lambda e: e.t0)


def similarity(query: dict, ev: Event) -> tuple[float, tuple[str, ...]]:
    """Weighted tag similarity: type 0.4, region 0.2, channel Jaccard 0.4."""
    reasons = []
    score = 0.0
    if query.get("type") == ev.type:
        score += 0.4
        reasons.append(f"유형 일치({ev.type})")
    if query.get("region") == ev.region:
        score += 0.2
        reasons.append(f"지역 일치({ev.region})")
    q = set(query.get("channels", []))
    if q or ev.channels:
        inter = q & ev.channels
        jac = len(inter) / len(q | ev.channels)
        score += 0.4 * jac
        if inter:
            reasons.append("공통 경로: " + ", ".join(sorted(inter)))
    return round(score, 4), tuple(reasons)


def find_analogs(
    query: dict,
    events: list[Event],
    cutoff: pd.Timestamp,
    k: int,
    min_similarity: float,
    exclude_id: str | None = None,
    same_type_only: bool = False,
    eligible: Callable[[Event], bool] | None = None,
) -> list[Analog]:
    """Top-k analogs that started before ``cutoff``, ranked by tag similarity.

    ``eligible`` lets the caller drop events whose observation window would overlap the
    forecast date (no look-ahead, see quant.scenario.analog_eligibility).
    """
    cands = []
    for ev in events:
        if ev.id == exclude_id or ev.t0 >= cutoff:
            continue
        if eligible is not None and not eligible(ev):
            continue
        if same_type_only and ev.type != query.get("type"):
            continue
        s, why = similarity(query, ev)
        if s >= min_similarity:
            cands.append((s, ev, why))
    cands.sort(key=lambda x: (-x[0], -x[1].t0.value))
    top = cands[:k]
    total = sum(s for s, _, _ in top) or 1.0
    return [Analog(ev, s, s / total, why) for s, ev, why in top]
