"""Declarative data-quality checks for the raw source series.

Modeled on pandera's ideas without the dependency (its checks are table-oriented, ours are
time-series specific):

- Schemas are data: each source series kind declares its checks once (``SCHEMAS``), and the
  same declaration documents the rules in the app.
- Lazy validation: every check runs and all failures are collected, never stop at the first.
- Failure cases: each failure lists how many observations broke the rule and sample dates.
- Severity: "error" (the series is likely wrong) or "warning" (worth a look). Validation never
  raises and never alters data, so the app keeps running; tools turn relevant failures into
  unknowns for the reviewer.

Thresholds are set above the largest genuine moves in the history (2008, 2020), e.g. the
biggest one-day ETF move is about 22% and the biggest VIX close-to-close change about 25pt.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable

import numpy as np
import pandas as pd

from sentinel.data.universe import ASSETS, FACTORS

MAX_SAMPLES = 5


@dataclass(frozen=True)
class Check:
    name: str
    description: str  # Korean, shown in the app as the rule's documentation
    fn: Callable[[pd.Series, "Context"], pd.Index]  # returns the dates that fail the rule
    severity: str = "error"  # "error" | "warning"
    structural: bool = False  # affects the whole series, whatever the analysis date


@dataclass(frozen=True)
class Context:
    ref_date: pd.Timestamp  # latest date seen across the price series (for staleness)


@dataclass
class Failure:
    source: str
    key: str
    check: str
    severity: str
    description: str
    n_cases: int
    dates: list[pd.Timestamp] = field(default_factory=list)  # sample dates (earliest first)
    structural: bool = False

    def relevant_at(self, base: pd.Timestamp) -> bool:
        """Whether the failure can affect an analysis using data up to ``base``."""
        return self.structural or (bool(self.dates) and self.dates[0] <= pd.Timestamp(base))

    def as_row(self) -> dict:
        return {"소스": self.source, "항목": self.key, "검사": self.check, "심각도": self.severity,
                "건수": self.n_cases, "예시 날짜": ", ".join(str(d.date()) for d in self.dates),
                "규칙": self.description}


@dataclass(frozen=True)
class SeriesSchema:
    name: str
    checks: tuple[Check, ...]

    def validate(self, s: pd.Series | None, source: str, key: str, ctx: Context) -> list[Failure]:
        out = []
        for c in self.checks:
            if s is None or s.empty:
                if c.name == "not_empty":
                    out.append(Failure(source, key, c.name, c.severity, c.description, 1, [], True))
                continue
            try:
                bad = c.fn(s, ctx)
            except Exception as exc:  # noqa: BLE001 - a broken check must not stop validation
                out.append(Failure(source, key, c.name, "warning", f"검사 실행 실패: {type(exc).__name__}", 1, [], True))
                continue
            if len(bad):
                bad = pd.DatetimeIndex(bad).sort_values()
                out.append(Failure(source, key, c.name, c.severity, c.description, len(bad),
                                   list(bad[:MAX_SAMPLES]), c.structural))
        return out


# ------------------------------------------------------------------------- check builders
def _none(_s, _c) -> pd.Index:
    return pd.DatetimeIndex([])


def not_empty() -> Check:
    return Check("not_empty", "데이터가 비어 있지 않아야 함", _none, "error", True)


def unique_sorted_index() -> Check:
    def fn(s, _c):
        idx = s.index
        dup = idx[idx.duplicated()]
        return dup if len(dup) or idx.is_monotonic_increasing else idx[:1]
    return Check("unique_sorted_index", "날짜가 중복 없이 오름차순이어야 함", fn, "error", True)


def no_future_dates() -> Check:
    def fn(s, _c):
        today = pd.Timestamp.now(tz="UTC").tz_localize(None).normalize()
        return s.index[s.index > today + pd.Timedelta(days=1)]
    return Check("no_future_dates", "미래 날짜가 없어야 함", fn, "error", True)


def in_range(lo: float, hi: float) -> Check:
    def fn(s, _c):
        v = s.dropna()
        return v.index[(v <= lo) | (v >= hi)]
    return Check("in_range", f"값이 정상 범위 안이어야 함 ({lo:g} 초과 {hi:g} 미만)", fn, "error")


def max_change(limit: float, transform: str) -> Check:
    def fn(s, _c):
        v = s.dropna()
        if transform == "logret":
            v = v[v > 0]  # non-positive values are the in_range check's job
        chg = np.log(v / v.shift(1)) if transform == "logret" else v.diff()
        return chg.index[chg.abs() > limit]
    unit = "로그수익률" if transform == "logret" else "변화폭"
    return Check("max_change", f"하루 {unit} 절댓값이 {limit:g} 이하여야 함 (역사적 최대치 초과 여부)", fn, "warning")


def spike_reversal(min_move: float = 0.05, sd_mult: float = 8.0, revert: float = 0.7) -> Check:
    """Bad tick pattern: a jump far outside recent volatility that reverses the next day."""
    def fn(s, _c):
        v = s.dropna()
        v = v[v > 0]
        r = np.log(v / v.shift(1))
        sd = r.rolling(60, min_periods=20).std().shift(1)
        nxt = r.shift(-1)
        m = (r.abs() > sd_mult * sd) & (r.abs() > min_move) & (nxt * r < 0) & (nxt.abs() > revert * r.abs())
        return r.index[m.fillna(False)]
    return Check("spike_reversal", "평소 변동성을 크게 넘는 급변 뒤 다음 날 되돌림(잘못된 시세 의심)이 없어야 함",
                 fn, "warning")


def max_gap(bdays: int) -> Check:
    def fn(s, _c):
        d = s.dropna().index.values.astype("datetime64[D]")
        if len(d) < 2:
            return pd.DatetimeIndex([])
        # Business days strictly between consecutive observations; reports the day the gap ends.
        missing = np.busday_count(d[:-1], d[1:]) - 1
        return pd.DatetimeIndex(d[1:][missing > bdays])
    return Check("max_gap", f"연속 결측 영업일이 {bdays}일 이하여야 함", fn, "warning")


def fresh(warn_bdays: int, error_bdays: int) -> list[Check]:
    def lag(s, c):
        last = s.dropna().index[-1]
        return len(pd.bdate_range(last, c.ref_date)) - 1, last

    def warn(s, c):
        n, last = lag(s, c)
        return pd.DatetimeIndex([last]) if warn_bdays < n <= error_bdays else pd.DatetimeIndex([])

    def err(s, c):
        n, last = lag(s, c)
        return pd.DatetimeIndex([last]) if n > error_bdays else pd.DatetimeIndex([])

    return [
        Check("stale", f"최신 날짜가 기준일보다 영업일 {warn_bdays}일 넘게 늦지 않아야 함", warn, "warning"),
        Check("stale_severe", f"최신 날짜가 기준일보다 영업일 {error_bdays}일 넘게 늦지 않아야 함", err, "error"),
    ]


def has_columns(cols: tuple[str, ...]) -> Check:
    def fn(df, _c):
        return pd.DatetimeIndex([]) if all(c in df for c in cols) else df.index[:1]
    return Check("has_columns", f"필수 열이 있어야 함: {', '.join(cols)}", fn, "error", True)


def frame_nonnegative() -> Check:
    def fn(df, _c):
        num = df.select_dtypes("number")
        return num.index[(num < 0).any(axis=1)]
    return Check("nonnegative", "지수 값이 음수가 아니어야 함", fn, "error")


# ------------------------------------------------------------------------------ schemas
_BASE = (not_empty(), unique_sorted_index(), no_future_dates())

PRICE_SCHEMA = SeriesSchema("ETF 수정종가", _BASE + (
    in_range(0.0, 1e6), max_change(0.30, "logret"), spike_reversal(), max_gap(5), *fresh(5, 20),
))

# Per-factor plausible ranges and one-day change limits. Fed H.10 is published weekly, so its
# series lag the market by up to about two weeks.
_FACTOR_RULES = {
    "oil": dict(lo=0.0, hi=400.0, chg=0.8, warn=5, err=20),
    "usd": dict(lo=50.0, hi=250.0, chg=0.05, warn=10, err=30),
    "us10y": dict(lo=-3.0, hi=25.0, chg=0.8, warn=5, err=20),
    "vix": dict(lo=5.0, hi=150.0, chg=35.0, warn=5, err=20),
    "krw": dict(lo=300.0, hi=5000.0, chg=0.2, warn=10, err=30),
}


def factor_schema(key: str) -> SeriesSchema:
    f, r = FACTORS[key], _FACTOR_RULES[key]
    return SeriesSchema(f.name, _BASE + (
        in_range(r["lo"], r["hi"]), max_change(r["chg"], f.transform), max_gap(10), *fresh(r["warn"], r["err"]),
    ))


GPR_SCHEMA = SeriesSchema("GPR 일간 지수", (
    not_empty(), has_columns(("GPRD", "GPRD_ACT", "GPRD_THREAT")), unique_sorted_index(),
    no_future_dates(), frame_nonnegative(), *fresh(10, 30),
))


def schema_table() -> pd.DataFrame:
    """The declared rules, for display as documentation."""
    rows = []
    for label, sch in [("ETF 시세", PRICE_SCHEMA)] + [(FACTORS[k].name, factor_schema(k)) for k in _FACTOR_RULES] \
            + [("GPR", GPR_SCHEMA)]:
        rows += [{"대상": label, "검사": c.name, "심각도": c.severity, "규칙": c.description} for c in sch.checks]
    return pd.DataFrame(rows)


# ----------------------------------------------------------------------------- validate
def validate_all(
    prices: dict[str, pd.Series], factors: dict[str, pd.Series], gpr: pd.DataFrame | None,
    ref_date: pd.Timestamp | None = None,
) -> list[Failure]:
    """Validate every raw series (lazy: all failures are collected)."""
    lasts = [s.dropna().index[-1] for s in prices.values() if s is not None and s.notna().any()]
    ctx = Context(pd.Timestamp(ref_date) if ref_date is not None else (max(lasts) if lasts else pd.Timestamp.now()))
    out: list[Failure] = []
    for t, a in ASSETS.items():
        if not a.synthetic:
            out += PRICE_SCHEMA.validate(prices.get(t), "yahoo", t, ctx)
    for k in _FACTOR_RULES:
        f = FACTORS[k]
        out += factor_schema(k).validate(factors.get(k), f.source, k, ctx)
    out += GPR_SCHEMA.validate(gpr, "gpr", "daily", ctx)
    return sorted(out, key=lambda x: (x.severity != "error", x.source, x.key))
