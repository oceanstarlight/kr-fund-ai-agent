"""Declarative data-quality checks: lazy, report-only, with failure cases."""

import numpy as np
import pandas as pd

from sentinel.data import quality
from sentinel.data.store import MarketData
from sentinel.tools import ToolContext, estimate_sensitivities

DAYS = pd.bdate_range("2015-01-01", "2020-12-31")


def _price():
    rng = np.random.default_rng(0)
    return pd.Series(100 * np.exp(np.cumsum(rng.normal(0, 0.01, len(DAYS)))), index=DAYS)


def _checks(s, ref=None):
    ctx = quality.Context(ref or DAYS[-1])
    return {f.check: f for f in quality.PRICE_SCHEMA.validate(s, "yahoo", "SPY", ctx)}


def test_clean_series_passes():
    assert _checks(_price()) == {}


def test_all_failures_are_collected_with_cases():
    s = _price()
    s.iloc[500] *= 1.5  # bad tick: jump then reversal
    s = s.drop(s.index[800:810])  # ten missing business days
    s.iloc[100] = -1.0
    got = _checks(s)
    assert {"spike_reversal", "max_gap", "in_range", "max_change"} <= set(got)
    assert got["spike_reversal"].dates[0] == DAYS[500]
    assert got["in_range"].severity == "error"


def test_staleness_and_empty():
    s = _price()
    assert _checks(s, DAYS[-1] + pd.Timedelta(days=14))["stale"].severity == "warning"
    assert _checks(s, DAYS[-1] + pd.Timedelta(days=60))["stale_severe"].severity == "error"
    assert _checks(None)["not_empty"].structural


def test_failure_relevance_respects_analysis_date():
    f = quality.Failure("yahoo", "SPY", "spike_reversal", "warning", "", 1, [pd.Timestamp("2020-01-02")])
    assert not f.relevant_at(pd.Timestamp("2019-12-31"))
    assert f.relevant_at(pd.Timestamp("2020-06-30"))


def test_quality_failures_become_digit_free_unknowns(synthetic_md):
    md = MarketData(synthetic_md.prices, synthetic_md.factors, None, synthetic_md.inception)
    md.quality = [quality.Failure("yahoo", "SPY", "in_range", "error", "", 1, [pd.Timestamp("2010-01-04")]),
                  quality.Failure("yahoo", "GLD", "max_gap", "warning", "", 1, [pd.Timestamp("2010-01-04")])]
    res = estimate_sensitivities(ToolContext(md, {"SPY": 1.0}, pd.Timestamp("2015-01-02")))
    codes = {u.code: u for u in res.unknowns}
    assert "data_quality_error" in codes and "data_quality_warning" not in codes  # GLD not held
    assert not any(ch.isdigit() for ch in codes["data_quality_error"].message)


def test_schema_table_documents_rules():
    t = quality.schema_table()
    assert {"stale", "spike_reversal", "has_columns"} <= set(t["검사"])


def test_tools_tolerate_market_data_without_quality(synthetic_md):
    """Objects cached by older code (no ``quality`` attribute) must not break analysis."""
    md = MarketData(synthetic_md.prices, synthetic_md.factors, None, synthetic_md.inception)
    del md.quality
    res = estimate_sensitivities(ToolContext(md, {"SPY": 1.0}, pd.Timestamp("2015-01-02")))
    assert not any(u.code.startswith("data_quality") for u in res.unknowns)
