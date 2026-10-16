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


def provider() -> str:
    """Active LLM provider: 'gemini', 'anthropic' or 'none'."""
    if os.environ.get("SENTINEL_DISABLE_LLM"):
        return "none"
    has_gemini = bool(os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY"))
    has_anthropic = bool(os.environ.get("ANTHROPIC_API_KEY") or os.environ.get("ANTHROPIC_AUTH_TOKEN"))
    choice = config.LLM_PROVIDER
    if choice == "gemini":
        return "gemini" if has_gemini else "none"
    if choice == "anthropic":
        return "anthropic" if has_anthropic else "none"
    if choice == "none":
        return "none"
    return "gemini" if has_gemini else "anthropic" if has_anthropic else "none"


def llm_available() -> bool:
    return provider() != "none"


def models() -> tuple[str, str]:
    """(fast model for classification, smart model for the brief) of the active provider."""
    if provider() == "gemini":
        return config.GEMINI_MODEL_FAST, config.GEMINI_MODEL_SMART
    return config.ANTHROPIC_MODEL_FAST, config.ANTHROPIC_MODEL_SMART


def _anthropic():
    import anthropic

    return anthropic.Anthropic(max_retries=2, timeout=60.0)


_GEMINI_CLIENTS: dict[str, object] = {}


def _with_retry(fn, *args, attempts: int = 3, delay: float = 2.0):
    """Retry transient provider errors (free tiers return 429/503 under load)."""
    import time

    for i in range(attempts):
        try:
            return fn(*args)
        except Exception as exc:  # noqa: BLE001
            transient = type(exc).__name__ in {"ServerError", "RateLimitError", "APIConnectionError",
                                                "InternalServerError", "OverloadedError"} or "429" in str(exc)
            if not transient or i == attempts - 1:
                raise
            time.sleep(delay * (2**i))


def _gemini():
    # Keep one client per key: a temporary client is closed when garbage-collected,
    # which breaks the request it was created for.
    from google import genai

    key = os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY") or ""
    if key not in _GEMINI_CLIENTS:
        _GEMINI_CLIENTS[key] = genai.Client(api_key=key)
    return _GEMINI_CLIENTS[key]


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


def _classify_anthropic(text: str) -> EventClassification | None:
    resp = _anthropic().messages.parse(
        model=models()[0],
        max_tokens=2048,
        system=CLASSIFY_SYSTEM,
        messages=[{"role": "user", "content": f"Event report:\n{text}"}],
        output_format=EventClassification,
    )
    if resp.stop_reason == "refusal":
        return None
    return resp.parsed_output


def _classify_gemini(text: str) -> EventClassification | None:
    from google.genai import types

    resp = _gemini().models.generate_content(
        model=models()[0],
        contents=f"Event report:\n{text}",
        config=types.GenerateContentConfig(
            system_instruction=CLASSIFY_SYSTEM,
            response_mime_type="application/json",
            response_schema=EventClassification,
            max_output_tokens=1024,
            temperature=0.0,
        ),
    )
    if isinstance(resp.parsed, EventClassification):
        return resp.parsed
    return EventClassification.model_validate_json(resp.text) if resp.text else None


def classify_event(text: str) -> tuple[EventClassification, str]:
    """Returns (classification, engine) where engine is 'llm:<provider>' or 'rules...'."""
    p = provider()
    if p == "none":
        return classify_rules(text), "rules"
    try:
        out = _with_retry(_classify_gemini if p == "gemini" else _classify_anthropic, text)
    except Exception as exc:  # network/API failure must not break the pipeline
        return classify_rules(text), f"rules ({p} error: {type(exc).__name__})"
    if out is None:
        return classify_rules(text), f"rules ({p} returned no result)"
    return out, f"llm:{p}"


# ---------------------------------------------------------------------------- brief
BRIEF_SYSTEM = (
    "You write the interpretation section of a Korean 'Impact Brief' for an institutional "
    "investor's risk team. Tables with the numbers are added by code; you only explain them.\n"
    "Rules:\n"
    "1. Output exactly one Markdown section starting with '### 해석' and 3-5 bullet points.\n"
    "2. Never type a digit. To mention a figure, write a placeholder [[key]] using only keys "
    "from FACTS. Horizons may be written as D+1, D+5 or D+20. Do not write labels that contain "
    "digits (no 'Q50', 'VaR95', '1위', '95%'); use words such as '기본 시나리오', '불리 시나리오', "
    "'과거 위험가치'.\n"
    "3. Do not add facts that are not in FACTS or CONTEXT. Explain which transmission channels "
    "drive the result, which assets dominate the loss, and how much to trust it.\n"
    "4. Be concise and factual. No investment advice."
)

_PLACEHOLDER = re.compile(r"\[\[([a-zA-Z0-9_]+)\]\]")
_ALLOWED_DIGITS = re.compile(r"D\+(?:1|5|20)\b|^\s*\d+\.\s", re.MULTILINE)
# Numbers spelled out in words would bypass the digit check.
_NUMBER_WORDS = re.compile(
    r"(?:[일이삼사오육칠팔구십백천만]+\s*(?:퍼센트|프로|배|분의))|퍼센트|절반|두\s*배|세\s*배|"
    r"\b(?:zero|one|two|three|four|five|six|seven|eight|nine|ten|twenty|thirty|fifty|hundred|"
    r"percent|half|double|triple)\b",
    re.IGNORECASE,
)
_NUMERIC_TOKEN = re.compile(r"[+-]?\d[\d,]*(?:\.\d+)?")


def audit_brief(raw: str, facts: dict[str, str]) -> list[str]:
    """Return a list of problems; empty means the brief may be published."""
    problems = []
    for key in _PLACEHOLDER.findall(raw):
        if key not in facts:
            problems.append(f"unknown fact key: {key}")
    if re.search(r"\]\]\s*(?:%|%p|bp|퍼센트)", raw):
        problems.append("unit appended to a placeholder")
    stripped = _ALLOWED_DIGITS.sub("", _PLACEHOLDER.sub("", raw))
    stray = re.findall(r"\d", stripped)
    if stray:
        problems.append(f"{len(stray)} digit(s) written outside placeholders")
    words = _NUMBER_WORDS.findall(stripped)
    if words:
        problems.append(f"number words outside placeholders: {', '.join(sorted(set(words)))[:80]}")
    if not problems:
        # After substitution every numeric token must come from a fact value, a horizon
        # label or a list marker.
        filled = _ALLOWED_DIGITS.sub("", fill_placeholders(raw, facts))
        allowed = {m for v in facts.values() for m in _NUMERIC_TOKEN.findall(v)}
        extra = [m for m in _NUMERIC_TOKEN.findall(filled) if m not in allowed]
        if extra:
            problems.append(f"numeric tokens not traceable to facts: {extra[:5]}")
    return problems


def fill_placeholders(raw: str, facts: dict[str, str]) -> str:
    return _PLACEHOLDER.sub(lambda m: facts.get(m.group(1), m.group(0)), raw)


def _brief_anthropic(payload: str) -> str | None:
    resp = _anthropic().messages.create(
        model=models()[1],
        max_tokens=4000,
        system=BRIEF_SYSTEM,
        messages=[{"role": "user", "content": payload}],
    )
    if resp.stop_reason == "refusal":
        return None
    return "".join(b.text for b in resp.content if b.type == "text")


def _brief_gemini(payload: str) -> str | None:
    from google.genai import types

    resp = _gemini().models.generate_content(
        model=models()[1],
        contents=payload,
        config=types.GenerateContentConfig(
            system_instruction=BRIEF_SYSTEM, max_output_tokens=4000, temperature=0.2
        ),
    )
    return resp.text


def write_brief_llm(facts: dict[str, str], context: dict) -> tuple[str | None, list[str]]:
    """Returns (filled brief or None, audit problems)."""
    p = provider()
    if p == "none":
        return None, ["llm unavailable"]
    payload = json.dumps({"FACTS": facts, "CONTEXT": context}, ensure_ascii=False, indent=1)
    try:
        raw = _with_retry(_brief_gemini if p == "gemini" else _brief_anthropic, payload)
    except Exception as exc:
        return None, [f"{p} error: {type(exc).__name__}"]
    if not raw:
        return None, [f"{p} returned no text"]
    problems = audit_brief(raw, facts)
    if problems:
        return None, problems
    return fill_placeholders(raw, facts), []
