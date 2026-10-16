"""LLM roles with deterministic fallbacks (Gemini or Claude).

Two calls per analyzed event:
- Planner: classifies the event and chooses analog-search parameters (structured output).
  On a failed rule check it re-plans with the verifier's feedback.
- Reviewer: reads the tool evidence and known unknowns, gives a confirm/caution verdict and
  writes the interpretation section.

LLMs never produce numbers. Text may reference figures only through ``[[fact_key]]``
placeholders, which are audited and substituted by code (see ``audit_brief``).
"""

from __future__ import annotations

import json
import os
import re
import threading
from contextlib import contextmanager
from typing import Literal

from pydantic import BaseModel, Field

from sentinel import config
from sentinel.tools import TOOL_CATALOG

TOOL_NAMES = TOOL_CATALOG

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


_LOCAL = threading.local()


@contextmanager
def disabled():
    """Turn the LLM off for the current thread only (e.g. automatic analyses in the app)."""
    prev = getattr(_LOCAL, "off", False)
    _LOCAL.off = True
    try:
        yield
    finally:
        _LOCAL.off = prev


def provider() -> str:
    """Active LLM provider: 'gemini', 'anthropic' or 'none'."""
    if os.environ.get("SENTINEL_DISABLE_LLM") or getattr(_LOCAL, "off", False):
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
class SearchParams(BaseModel):
    analog_k: int = Field(description="How many historical analogs to use, 3-10")
    min_similarity: float = Field(description="Minimum tag similarity for an analog, 0.05-0.6")
    same_type_only: bool = Field(description="Restrict analogs to the same event type")
    label_ko: str = Field(description="Short Korean label, e.g. '좁은 검색' or '넓은 검색'")


class Plan(BaseModel):
    classification: EventClassification
    candidates: list[SearchParams] = Field(
        description="One or two analog-search settings in order of preference (narrow first). "
        "Both are executed in parallel and the verifier keeps the first that passes."
    )
    rationale_ko: str = Field(description="1-2 Korean sentences on why this plan")


class Review(BaseModel):
    verdict: Literal["confirm", "caution"]
    concerns_ko: list[str] = Field(description="Up to 3 short Korean concerns; empty if none")
    interpretation_md: str = Field(description="Markdown section starting with '### 해석' and 3-5 bullets")


PLANNER_SYSTEM = (
    "You are the planner of a geopolitical/macro risk agent for institutional investors. "
    "Given an event report (and, when present, curated tags or verifier feedback), output a "
    "plan: classify the event with the allowed labels, and give one or two analog-search "
    "settings for the deterministic tools, preferred one first. A good default is a narrow "
    "setting (fewer, closer analogs) followed by a wider fallback; both run in parallel, so a "
    "second candidate avoids a costly re-plan. When verifier feedback says analogs are too few "
    "or the two engines disagree, widen both candidates. Write Korean text fields in Korean."
)

REVIEWER_SYSTEM = (
    "You are the final reviewer of a risk agent's output for an institutional investor. "
    "You receive FACTS (figures, referenced only by key), the event classification, the rule "
    "checks, and the tools' known unknowns. Tasks: (1) verdict 'caution' if the unknowns or "
    "failed checks materially limit the result, else 'confirm'; (2) up to three concerns; "
    "(3) an interpretation section.\n"
    "Rules for all text: never type a digit. Mention a figure only as a placeholder [[key]] "
    "using keys from FACTS. Horizons may be written as D+1, D+5 or D+20. Do not write labels "
    "with digits (no 'Q50', 'VaR95', '1위', '95%'); use words such as '기본 시나리오', "
    "'불리 시나리오', '과거 위험가치'. Describe estimates as expectations, not observations. "
    "Do not add facts that are not given. No investment advice. Write in Korean."
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


def _structured_anthropic(schema, system: str, user: str, smart: bool):
    resp = _anthropic().messages.parse(
        model=models()[1 if smart else 0],
        max_tokens=4000,
        system=system,
        messages=[{"role": "user", "content": user}],
        output_format=schema,
    )
    if resp.stop_reason == "refusal":
        return None
    return resp.parsed_output


def _structured_gemini(schema, system: str, user: str, smart: bool):
    from google.genai import types

    resp = _gemini().models.generate_content(
        model=models()[1 if smart else 0],
        contents=user,
        config=types.GenerateContentConfig(
            system_instruction=system,
            response_mime_type="application/json",
            response_schema=schema,
            max_output_tokens=3000,
            temperature=0.1,
        ),
    )
    if isinstance(resp.parsed, schema):
        return resp.parsed
    return schema.model_validate_json(resp.text) if resp.text else None


_CACHE = None


def _cache():
    global _CACHE
    if _CACHE is None and config.LLM_CACHE:
        try:
            import diskcache

            _CACHE = diskcache.Cache(str(config.LLM_CACHE_DIR), size_limit=200 * 2**20)
        except Exception:  # noqa: BLE001 - caching is an optimization only
            _CACHE = False
    # An empty diskcache.Cache is falsy (it defines __len__), so test identity, not truthiness.
    return None if _CACHE is None or _CACHE is False else _CACHE


def _cache_key(p: str, schema, system: str, user: str, smart: bool) -> str:
    import hashlib

    raw = json.dumps([p, models()[1 if smart else 0], schema.__name__,
                      json.dumps(schema.model_json_schema(), sort_keys=True), system, user], ensure_ascii=False)
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


def structured_call(schema, system: str, user: str, smart: bool = False):
    """Returns (parsed object or None, error string or None). Successful results are cached."""
    p = provider()
    if p == "none":
        return None, "llm unavailable"
    cache = _cache()
    key = _cache_key(p, schema, system, user, smart) if cache is not None else None
    if cache is not None and key in cache:
        try:
            return schema.model_validate_json(cache[key]), None
        except Exception:  # noqa: BLE001 - stale entry from an older schema
            del cache[key]
    fn = _structured_gemini if p == "gemini" else _structured_anthropic
    try:
        out = _with_retry(fn, schema, system, user, smart)
    except Exception as exc:  # network/API failure must not break the pipeline
        return None, f"{p} error: {type(exc).__name__}"
    if out is None:
        return None, f"{p} returned no result"
    if cache is not None:
        cache[key] = out.model_dump_json()
    return out, None


# ----------------------------------------------------------------------------- planner
def plan_rules(text: str, attempt: int, tags: dict | None = None) -> Plan:
    """Deterministic planner: curated tags or keyword rules, a narrow and a wide candidate."""
    cls = EventClassification(**{**tags, "headline_ko": tags.get("headline_ko", text[:80]),
                                 "rationale_ko": tags.get("rationale_ko", "")}) if tags else classify_rules(text)
    if attempt == 0:
        cands = [
            SearchParams(analog_k=config.ANALOG_K, min_similarity=config.ANALOG_MIN_SIMILARITY,
                         same_type_only=False, label_ko="기본 검색"),
            SearchParams(analog_k=config.ANALOG_K + 2, min_similarity=0.15, same_type_only=True, label_ko="동일 유형 확장"),
        ]
    else:
        cands = [SearchParams(analog_k=config.ANALOG_K + 5, min_similarity=0.05, same_type_only=False, label_ko="최대 확장")]
    return Plan(classification=cls, candidates=cands, rationale_ko=f"규칙 기반 계획 (시도 {attempt + 1})")


def plan_event(text: str, attempt: int = 0, tags: dict | None = None, feedback: list[str] | None = None) -> tuple[Plan, str]:
    """Returns (plan, engine). Curated tags, when given, override the LLM's classification."""
    if not llm_available():
        return plan_rules(text, attempt, tags), "rules"
    user = json.dumps(
        {"event_report": text, "curated_tags": tags, "attempt": attempt + 1,
         "verifier_feedback": feedback or [], "tools": TOOL_NAMES},
        ensure_ascii=False,
    )
    plan, err = structured_call(Plan, PLANNER_SYSTEM, user, smart=False)
    if plan is None or not plan.candidates:
        return plan_rules(text, attempt, tags), f"rules ({err or 'empty plan'})"
    plan.candidates = plan.candidates[:2]
    if tags:  # replay: keep the curated ground-truth tags, let the LLM choose search parameters
        plan.classification = EventClassification(**{**plan.classification.model_dump(), **{
            k: tags[k] for k in ("type", "region", "channels")}})
    return plan, f"llm:{provider()}"


def classify_event(text: str) -> tuple[EventClassification, str]:
    plan, engine = plan_event(text)
    return plan.classification, engine


# ---------------------------------------------------------------------------- reviewer
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


def _normalize_md(md: str) -> str:
    """Some models return bullets on one line ('### 해석* a* b'); put each bullet on its own line."""
    md = md.strip()
    if "\n" in md or "**" in md:  # already multi-line, or bold markers we must not split
        return md
    md = re.sub(r"^(#{1,6} [^*-]+?)\s*(?=[*-] )", r"\1\n", md)
    return re.sub(r"(?<!\n)\s*\*\s+(?=\S)", "\n- ", md)


def review_result(facts: dict[str, str], context: dict) -> tuple[Review | None, list[str]]:
    """Final LLM review. Returns (review or None, problems). Text is audited and filled."""
    if not llm_available():
        return None, ["llm unavailable"]
    payload = json.dumps({"FACTS": facts, "CONTEXT": context}, ensure_ascii=False, indent=1)
    review, err = structured_call(Review, REVIEWER_SYSTEM, payload, smart=True)
    if review is None:
        return None, [err or "no result"]
    text = review.interpretation_md + "\n" + "\n".join(review.concerns_ko)
    problems = audit_brief(text, facts)
    if problems:
        return None, problems
    review.interpretation_md = _normalize_md(fill_placeholders(review.interpretation_md, facts))
    review.concerns_ko = [fill_placeholders(c, facts) for c in review.concerns_ko[:3]]
    return review, []
