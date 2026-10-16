"""Leave-one-out backtest over the event library.

For each event: forecast with data up to the pre-event close, analogs restricted to events
whose windows closed before that date (and never the event itself), then compare with the
realized portfolio return. LLMs are not used, so results are deterministic.
"""

from __future__ import annotations

import os

import numpy as np
import pandas as pd

from sentinel import config
from sentinel.agents.pipeline import run
from sentinel.quant.engines import realized
from sentinel.data.store import MarketData
from sentinel.events.library import load_events


def run_backtest(
    md: MarketData, weights: dict[str, float], horizons=config.HORIZONS, vol_scale: bool = True
) -> pd.DataFrame:
    prev = os.environ.get("SENTINEL_DISABLE_LLM")
    os.environ["SENTINEL_DISABLE_LLM"] = "1"
    rows = []
    try:
        for ev in load_events():
            if md.trading_day_before(ev.t0) is None or md.window(ev.t0, max(horizons)) is None:
                continue
            st = run(md, weights, mode="replay", event_id=ev.id, horizons=tuple(horizons), vol_scale=vol_scale)
            for h in horizons:
                p = st["impacts"][h]
                r = st["realized"].get(h)
                if r is None or p.n_analogs == 0:
                    continue
                rows.append(
                    {
                        "event": ev.id,
                        "name": ev.name,
                        "t0": ev.t0.date(),
                        "horizon": h,
                        "n_analogs": p.n_analogs,
                        "grade": st["verification"]["grade"],
                        "detected": st["detection"]["triggered"],
                        "pred_a": p.engine_a["q50"],
                        "pred_b": p.engine_b["q50"],
                        "pred": p.blend["q50"],
                        "q25": p.blend["q25"],
                        "q75": p.blend["q75"],
                        "realized": r["realized"],
                    }
                )
    finally:
        if prev is None:
            os.environ.pop("SENTINEL_DISABLE_LLM", None)
        else:
            os.environ["SENTINEL_DISABLE_LLM"] = prev
    df = pd.DataFrame(rows)
    if not df.empty:
        df["shock_baseline"] = _shock_baseline(md, weights, df)
    return df


def _shock_baseline(md: MarketData, weights: dict[str, float], df: pd.DataFrame) -> pd.Series:
    """Naive 'shock event' forecast: median realized return of all earlier library events.

    Same eligibility rule as the model (window closed before the forecast base), but no
    similarity and no volatility scaling. Separates event-specific skill from the generic
    fact that curated shock events tend to be down days.
    """
    events = {e.id: e for e in load_events()}
    max_h = int(df["horizon"].max())
    out = []
    for _, r in df.iterrows():
        base = md.trading_day_before(events[r["event"]].t0)
        past = []
        for e in events.values():
            w_max = md.window(e.t0, max_h)
            if e.id == r["event"] or w_max is None or w_max[1] >= base:
                continue
            rr = realized(md, weights, e.t0, int(r["horizon"]))
            if rr is not None:
                past.append(rr["portfolio"])
        out.append(float(np.median(past)) if past else np.nan)
    return pd.Series(out, index=df.index)


def _spearman(a: pd.Series, b: pd.Series) -> float:
    if len(a) < 3:
        return float("nan")
    return float(a.rank().corr(b.rank()))


def summarize(df: pd.DataFrame) -> pd.DataFrame:
    out = []
    for h, g in df.groupby("horizon"):
        err = (g["pred"] - g["realized"]).abs()
        out.append(
            {
                "horizon": f"D+{h}",
                "events": len(g),
                "direction_hit": float(((g["pred"] >= 0) == (g["realized"] >= 0)).mean()),
                "mae_blend": float(err.mean()),
                "mae_engine_a": float((g["pred_a"] - g["realized"]).abs().mean()),
                "mae_engine_b": float((g["pred_b"] - g["realized"]).abs().mean()),
                "mae_zero_baseline": float(g["realized"].abs().mean()),
                "mae_shock_baseline": float((g["shock_baseline"] - g["realized"]).abs().mean()),
                "direction_hit_shock_baseline": float(((g["shock_baseline"] >= 0) == (g["realized"] >= 0)).mean()),
                "share_negative": float((g["realized"] < 0).mean()),
                "skill_vs_zero": float(1 - err.mean() / g["realized"].abs().mean()) if g["realized"].abs().mean() else np.nan,
                "iqr_coverage": float(((g["realized"] >= g["q25"]) & (g["realized"] <= g["q75"])).mean()),
                "rank_corr": _spearman(g["pred"], g["realized"]),
                "detected_rate": float(g["detected"].mean()),
            }
        )
    return pd.DataFrame(out)


def to_markdown(
    summary: pd.DataFrame, detail: pd.DataFrame, title: str, ablation: pd.DataFrame | None = None
) -> str:
    pct = lambda x: f"{x:+.2%}"  # noqa: E731
    lines = [
        f"# 백테스트 결과: {title}",
        "",
        "- 방식: 사건별 Leave-one-out. 예측은 사건 직전 종가까지의 데이터만 사용, 유사 사건은 관측 구간이 예측 기준일 이전에 끝난 사건만 사용",
        "- 기준선: '변화 없음(0%)' 예측. skill_vs_zero > 0 이면 기준선보다 오차가 작음",
        "- 충격 기준선: 과거 라이브러리 사건 전체의 실측 중앙값(유사도·변동성 보정 없음). 이 기준선과 비교해야 '충격 사건은 대체로 하락' 이상의 사건별 판별력을 볼 수 있음",
        "- iqr_coverage: 실측이 예측 구간(하위 25%~상위 25%) 안에 들어온 비율. 잘 보정된 경우 약 50%",
        "- 주의: 사건 라이브러리는 사후적으로 '시장에 영향을 준 사건'을 골라 만든 것이라 선택 편향이 있음. 사건 수가 적고 일부 기간이 겹침",
        "",
        "## 요약",
        "| 기간 | 사건 수 | 실측 하락 비율 | 방향 적중 | 방향 적중(충격 기준선) | MAE(통합) | MAE(엔진A) | MAE(엔진B) | MAE(0% 기준선) | MAE(충격 기준선) | 0% 기준선 대비 개선 | 구간 포함률 | 순위상관 | 감지율 |",
        "|---|---|---|---|---|---|---|---|---|---|---|---|---|---|",
    ]
    for _, r in summary.iterrows():
        lines.append(
            f"| {r['horizon']} | {r['events']} | {r['share_negative']:.0%} | {r['direction_hit']:.0%} | {r['direction_hit_shock_baseline']:.0%} "
            f"| {r['mae_blend']:.2%} | {r['mae_engine_a']:.2%} | {r['mae_engine_b']:.2%} | {r['mae_zero_baseline']:.2%} "
            f"| {r['mae_shock_baseline']:.2%} | {r['skill_vs_zero']:+.0%} | {r['iqr_coverage']:.0%} "
            f"| {r['rank_corr']:+.2f} | {r['detected_rate']:.0%} |"
        )
    if ablation is not None:
        lines += [
            "",
            "## 비교 실험: 변동성 국면 보정 미적용",
            "유사 사건의 변동을 현재 변동성 국면에 맞춰 조정하지 않았을 때의 결과(본 결과와 비교용).",
            "",
            "| 기간 | 방향 적중 | MAE(통합) | MAE(0% 기준선) | 기준선 대비 개선 | 구간 포함률 | 순위상관 |",
            "|---|---|---|---|---|---|---|",
        ]
        for _, r in ablation.iterrows():
            lines.append(
                f"| {r['horizon']} | {r['direction_hit']:.0%} | {r['mae_blend']:.2%} | {r['mae_zero_baseline']:.2%} "
                f"| {r['skill_vs_zero']:+.0%} | {r['iqr_coverage']:.0%} | {r['rank_corr']:+.2f} |"
            )
    lines += ["", "## 사건별 결과", "| 사건 | t0 | 기간 | 유사 사건 수 | 등급 | 예측 | 구간 | 실측 |", "|---|---|---|---|---|---|---|---|"]
    for _, r in detail.iterrows():
        lines.append(
            f"| {r['name']} | {r['t0']} | D+{r['horizon']} | {r['n_analogs']} | {r['grade']} | {pct(r['pred'])} "
            f"| {pct(r['q25'])} ~ {pct(r['q75'])} | {pct(r['realized'])} |"
        )
    return "\n".join(lines) + "\n"
