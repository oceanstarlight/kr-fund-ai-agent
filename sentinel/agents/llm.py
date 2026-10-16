"""Claude calls with deterministic fallbacks.

LLMs only classify events and write prose. Every number shown to the user comes from the
quant tools; the brief writer must reference numbers through ``[[fact_key]]`` placeholders,
which are substituted afterwards and audited (see ``audit_brief``).
"""

from __future__ import annotations

import json
import os
import re
from typing import Literal

from pydantic import BaseModel, Field

from sentinel import config

EventType = Literal[
    "war", "military_strike", "trade", "energy_supply", "financial_stress",
    "political", "pandemic", "disaster", "fx_shock",
]
Region = Literal["middle_east", "europe", "east_asia", "korea", "north_america", "emerging", "global"]
Channel = Literal[
    "oil", "risk_off", "safe_haven", "trade", "supply_chain", "banking", "rates",
    "em_fx", "krw", "china", "europe", "japan", "usd",
]


class EventClassification(BaseModel):
    type: EventType
    region: Region
    channels: list[Channel] = Field(description="2-5 transmission channels to markets")
    headline_ko: str = Field(description="One-line Korean headline of the event")
    rationale_ko: str = Field(description="1-2 Korean sentences on why these channels apply")


def llm_available() -> bool:
    if os.environ.get("SENTINEL_DISABLE_LLM"):
        return False
    return bool(os.environ.get("ANTHROPIC_API_KEY") or os.environ.get("ANTHROPIC_AUTH_TOKEN"))


def _client():
    import anthropic

    return anthropic.Anthropic(max_retries=2, timeout=60.0)


# --------------------------------------------------------------------- classification
CLASSIFY_SYSTEM = (
    "You classify geopolitical and macro events for an institutional investor's risk monitor. "
    "Pick the event type, the primary region, and the market transmission channels that a "
    "portfolio manager would expect to matter in the first weeks. Use only the allowed labels. "
    "Write headline_ko and rationale_ko in Korean."
)

_RULES: list[tuple[str, dict]] = [
    (r"invasion|invade|war\b|전쟁|침공", {"type": "war", "channels": ["risk_off", "safe_haven"]}),
    (r"strike|missile|drone|attack|airstrike|공습|미사일|피격|사살|군사", {"type": "military_strike", "channels": ["risk_off", "safe_haven"]}),
    (r"blockade|봉쇄", {"type": "military_strike", "channels": ["risk_off", "supply_chain", "trade"]}),
    (r"tariff|trade war|관세|무역", {"type": "trade", "channels": ["trade", "risk_off", "supply_chain"]}),
    (r"oil|crude|opec|refiner|pipeline|원유|유가|석유", {"type": "energy_supply", "channels": ["oil", "supply_chain"]}),
    (r"bank|default|bankrupt|downgrade|은행|파산|강등|부도", {"type": "financial_stress", "channels": ["banking", "risk_off", "safe_haven"]}),
    (r"election|referendum|martial law|coup|impeach|선거|국민투표|계엄|탄핵", {"type": "political", "channels": ["risk_off"]}),
    (r"pandemic|virus|outbreak|팬데믹|감염병", {"type": "pandemic", "channels": ["risk_off", "safe_haven", "supply_chain"]}),
    (r"earthquake|tsunami|hurricane|지진|쓰나미|태풍", {"type": "disaster", "channels": ["supply_chain", "risk_off"]}),
    (r"devalu|currency|yen|yuan|carry|환율|평가절하|엔화|위안", {"type": "fx_shock", "channels": ["em_fx", "risk_off"]}),
]
_REGIONS: list[tuple[str, str]] = [
    (r"korea|seoul|pyongyang|한국|북한|서울", "korea"),
    (r"iran|israel|saudi|gaza|houthi|red sea|iraq|이란|이스라엘|사우디|홍해|중동", "middle_east"),
    (r"china|japan|taiwan|yuan|yen|중국|일본|대만", "east_asia"),
    (r"russia|ukraine|europe|euro|uk\b|britain|러시아|우크라이나|유럽|영국", "europe"),
    (r"united states|\bus\b|fed\b|washington|미국|연준", "north_america"),
]
_EXTRA_CHANNELS: list[tuple[str, str]] = [
    (r"oil|crude|energy|iran|saudi|hormuz|red sea|유가|원유|에너지|호르무즈", "oil"),
    (r"korea|krw|won\b|한국|원화|북한", "krw"),
    (r"china|yuan|중국|위안", "china"),
    (r"japan|yen|일본|엔화", "japan"),
    (r"europe|euro|ukraine|russia|유럽|우크라이나|러시아", "europe"),
    (r"ship|supply|chip|semiconductor|공급망|반도체|해운", "supply_chain"),
    (r"rate|yield|treasury|금리|국채", "rates"),
]


def classify_rules(text: str) -> EventClassification:
    t = text.lower()
    found = {"type": "political", "channels": ["risk_off"]}
    for pat, val in _RULES:
        if re.search(pat, t):
            found = val
            break
    region = next((r for pat, r in _REGIONS if re.search(pat, t)), "global")
    channels = list(dict.fromkeys(found["channels"] + [c for pat, c in _EXTRA_CHANNELS if re.search(pat, t)]))
    return EventClassification(
        type=found["type"],
        region=region,
        channels=channels[:5],
        headline_ko=text.strip().splitlines()[0][:80],
        rationale_ko="키워드 규칙 기반 분류(LLM 미사용).",
    )


def classify_event(text: str) -> tuple[EventClassification, str]:
    """Returns (classification, engine) where engine is 'llm' or 'rules'."""
    if not llm_available():
        return classify_rules(text), "rules"
    try:
        resp = _client().messages.parse(
            model=config.MODEL_FAST,
            max_tokens=2048,
            system=CLASSIFY_SYSTEM,
            messages=[{"role": "user", "content": f"Event report:\n{text}"}],
            output_format=EventClassification,
        )
        if resp.stop_reason == "refusal" or resp.parsed_output is None:
            return classify_rules(text), "rules"
        return resp.parsed_output, "llm"
    except Exception:  # network/API failure must not break the pipeline
        return classify_rules(text), "rules"


# ---------------------------------------------------------------------------- brief
BRIEF_SYSTEM = (
    "You write a one-page Korean 'Impact Brief' for an institutional investor's risk team. "
    "Rules: (1) Never write a number yourself. Every figure, date or count must be written as "
    "a placeholder [[key]] using only keys from the provided FACTS. Horizon labels D+1, D+5, "
    "D+20 are allowed as text. (2) Do not add facts that are not in FACTS or CONTEXT. "
    "(3) Use Markdown with these sections: '### 요약', '### 예상 영향', '### 주요 기여 자산', "
    "'### 근거와 신뢰도', '### 유의사항'. (4) Be concise, factual, no investment advice."
)

_PLACEHOLDER = re.compile(r"\[\[([a-zA-Z0-9_]+)\]\]")
_ALLOWED_DIGITS = re.compile(r"D\+\d+")


def audit_brief(raw: str, facts: dict[str, str]) -> list[str]:
    """Return a list of problems; empty means the brief may be published."""
    problems = []
    for key in _PLACEHOLDER.findall(raw):
        if key not in facts:
            problems.append(f"unknown fact key: {key}")
    stripped = _ALLOWED_DIGITS.sub("", _PLACEHOLDER.sub("", raw))
    stray = re.findall(r"\d", stripped)
    if stray:
        problems.append(f"{len(stray)} digit(s) written outside placeholders")
    return problems


def fill_placeholders(raw: str, facts: dict[str, str]) -> str:
    return _PLACEHOLDER.sub(lambda m: facts.get(m.group(1), m.group(0)), raw)


def write_brief_llm(facts: dict[str, str], context: dict) -> tuple[str | None, list[str]]:
    """Returns (filled brief or None, audit problems)."""
    if not llm_available():
        return None, ["llm unavailable"]
    payload = json.dumps({"FACTS": facts, "CONTEXT": context}, ensure_ascii=False, indent=1)
    try:
        resp = _client().messages.create(
            model=config.MODEL_SMART,
            max_tokens=4000,
            system=BRIEF_SYSTEM,
            messages=[{"role": "user", "content": payload}],
        )
    except Exception as exc:
        return None, [f"api error: {type(exc).__name__}"]
    if resp.stop_reason == "refusal":
        return None, ["refusal"]
    raw = "".join(b.text for b in resp.content if b.type == "text")
    problems = audit_brief(raw, facts)
    if problems:
        return None, problems
    return fill_placeholders(raw, facts), []
