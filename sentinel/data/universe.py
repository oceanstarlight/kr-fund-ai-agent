"""Asset universe, macro factors and portfolio presets.

All assets are listed ETF proxies. Alternative assets (private equity, real estate,
infrastructure, hedge funds, private credit) are represented by listed proxies, so the
results describe the listed-market exposure of those sleeves, not appraisal values.
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class Asset:
    ticker: str
    name: str
    asset_class: str
    region: str
    # Used before the ETF's inception date: {ticker: weight}. Documented in the data manifest.
    backfill: dict[str, float] = field(default_factory=dict)
    synthetic: bool = False


ASSETS: dict[str, Asset] = {
    a.ticker: a
    for a in [
        Asset("SPY", "미국 주식", "주식", "북미"),
        Asset("EFA", "선진국 주식(미국 제외)", "주식", "선진국"),
        Asset("EEM", "신흥국 주식", "주식", "신흥국"),
        Asset("EWY", "한국 주식", "주식", "한국", backfill={"EEM": 1.0}),
        Asset("IEF", "미국 중기 국채", "채권", "북미"),
        Asset("TLT", "미국 장기 국채", "채권", "북미", backfill={"IEF": 1.0}),
        Asset("LQD", "미국 투자등급 회사채", "채권", "북미", backfill={"IEF": 1.0}),
        Asset("TIP", "미국 물가연동채", "채권", "북미", backfill={"IEF": 1.0}),
        Asset("EMB", "신흥국 달러채", "채권", "신흥국", backfill={"LQD": 0.6, "EEM": 0.2, "IEF": 0.2}),
        Asset("HYG", "미국 하이일드", "채권", "북미", backfill={"LQD": 0.7, "SPY": 0.3}),
        Asset("GLD", "금", "원자재", "글로벌"),
        Asset("PSP", "사모주식 프록시(상장 PE)", "대체", "글로벌", backfill={"SPY": 1.0}),
        Asset("VNQ", "부동산 프록시(리츠)", "대체", "북미"),
        Asset("IGF", "인프라 프록시(상장 인프라)", "대체", "글로벌", backfill={"SPY": 0.5, "EFA": 0.5}),
        Asset("QAI", "헤지펀드 프록시(멀티전략)", "대체", "글로벌", backfill={"SPY": 0.3, "IEF": 0.7}),
        Asset("BKLN", "사모채권 프록시(뱅크론)", "대체", "북미", backfill={"HYG": 1.0}),
        Asset("BIL", "단기자금(1~3개월 T-bill)", "현금", "북미", backfill={"IEF": 0.1}),
        Asset("KRW_CASH", "원화 자산 프록시(원/달러 환율만 반영)", "채권", "한국", synthetic=True),
    ]
}


@dataclass(frozen=True)
class Factor:
    key: str
    source: str  # "eia" | "fed" | "cboe" | "derived"
    source_id: str
    name: str
    transform: str  # "logret" or "diff"
    unit: str


# Primary (official) sources instead of FRED, whose terms restrict AI/ML-related use.
# The credit factor is derived from ETF prices (Moody's Baa yields are licensed data).
FACTORS: dict[str, Factor] = {
    f.key: f
    for f in [
        Factor("oil", "eia", "RBRTE", "브렌트유", "logret", "%"),
        Factor("usd", "fed", "JRXWTFB_N.B", "달러 지수(광의)", "logret", "%"),
        Factor("us10y", "fed", "RIFLGFCY10_N.B", "미 10년물 금리", "diff", "%p"),
        Factor("credit", "derived", "IEF/HYG", "신용 스트레스(국채 대비 하이일드 약세)", "logret", "%"),
        Factor("vix", "cboe", "VIX", "VIX", "diff", "pt"),
        Factor("krw", "fed", "RXI_N.B.KO", "원/달러 환율", "logret", "%"),
    ]
}
FACTOR_KEYS = list(FACTORS)


@dataclass(frozen=True)
class Preset:
    key: str
    name: str
    source: str
    weights: dict[str, float]
    note: str = ""


def _split(total: float, parts: dict[str, float]) -> dict[str, float]:
    return {k: total * v for k, v in parts.items()}


def _merge(*dicts: dict[str, float]) -> dict[str, float]:
    out: dict[str, float] = {}
    for d in dicts:
        for k, v in d.items():
            out[k] = out.get(k, 0.0) + v
    return {k: v / 100.0 for k, v in out.items()}


PRESETS: dict[str, Preset] = {
    "sovereign": Preset(
        "sovereign",
        "국부펀드형 (KIC 2025년 말 공개 배분)",
        "KIC 홈페이지 투자운용현황 (2025년 말)",
        _merge(
            _split(41.6, {"SPY": 0.60, "EFA": 0.28, "EEM": 0.12}),
            _split(32.8, {"IEF": 0.55, "LQD": 0.25, "TIP": 0.10, "EMB": 0.10}),
            {"PSP": 7.6, "VNQ": 5.0, "IGF": 4.5, "QAI": 2.6, "BKLN": 2.0, "BIL": 3.9},
        ),
        "자산군 비중은 공개값, 자산군 내부 지역·종목 분할은 가정",
    ),
    "pension": Preset(
        "pension",
        "연기금형 (국민연금 2026년 7월 말 공개 배분)",
        "국민연금기금운용본부 포트폴리오 현황 (2026년 7월 말)",
        _merge(
            _split(37.0, {"SPY": 0.65, "EFA": 0.25, "EEM": 0.10}),
            {"EWY": 24.9, "KRW_CASH": 16.8},
            _split(14.8, {"PSP": 1 / 3, "VNQ": 1 / 3, "IGF": 1 / 3}),
            _split(6.1, {"IEF": 0.65, "LQD": 0.35}),
            {"BIL": 0.4},
        ),
        "USD 기준 평가. 국내채권은 원화 환율 효과만 반영하는 프록시",
    ),
    "balanced": Preset(
        "balanced",
        "60/40 글로벌 균형형",
        "일반적 60/40 배분",
        _merge({"SPY": 36, "EFA": 18, "EEM": 6, "IEF": 30, "LQD": 10}),
    ),
}


def normalize(weights: dict[str, float]) -> dict[str, float]:
    unknown = [t for t in weights if t not in ASSETS]
    if unknown:
        raise ValueError(f"Unknown tickers: {unknown}")
    total = sum(weights.values())
    if total <= 0:
        raise ValueError("Portfolio weights must sum to a positive number")
    return {k: v / total for k, v in weights.items() if v != 0}
