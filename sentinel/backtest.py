"""Leave-one-out backtest over the event library.

For each event: forecast with data up to the pre-event close, analogs restricted to events
whose windows closed before that date (and never the event itself), then compare with the
realized portfolio return. LLMs are not used, so results are deterministic.
"""

from __future__ import annotations

import multiprocessing as mp
import os
from concurrent.futures import ProcessPoolExecutor

import numpy as np
import pandas as pd

from sentinel import config
from sentinel.agents.pipeline import run
from sentinel.quant.engines import realized
from sentinel.data.store import MarketData
from sentinel.events.library import load_events


_SHARED: dict = {}  # market data handed to forked workers without pickling it per job


def _replay_one(args) -> list[dict]:
    weights, ev_id, horizons, opts = args
    st = run(_SHARED["md"], weights, mode="replay", event_id=ev_id, horizons=tuple(horizons), **opts)
    ev = st["event"]
    rows = []
    for h in horizons:
        p = st["impacts"][h]
        r = st["realized"].get(h)
        if r is None or p.n_analogs == 0:
            continue
        rows.append(
            {
                "event": ev.id, "name": ev.name, "t0": ev.t0.date(), "horizon": h, "selection": ev.selection,
                "n_analogs": p.n_analogs, "grade": st["verification"]["grade"],
                "detected": st["detection"]["triggered"],
                "pred_a": p.engine_a["q50"], "pred_b": p.engine_b["q50"], "pred": p.blend["q50"],
                "q25": p.blend["q25"], "q75": p.blend["q75"], "realized": r["realized"],
            }
        )
    return rows


def run_backtest(
    md: MarketData, weights: dict[str, float], horizons=config.HORIZONS, vol_scale: bool = True,
    vol_method: str = config.VOL_METHOD, interval_method: str = config.INTERVAL_METHOD, workers: int | None = None,
) -> pd.DataFrame:
    """Events are independent, so they run in parallel processes (LLM disabled, deterministic)."""
    prev = os.environ.get("SENTINEL_DISABLE_LLM")
    os.environ["SENTINEL_DISABLE_LLM"] = "1"
    opts = {"vol_scale": vol_scale, "vol_method": vol_method, "interval_method": interval_method}
    _SHARED["md"] = md
    jobs = [
        (weights, ev.id, tuple(horizons), opts)
        for ev in load_events()
        if md.trading_day_before(ev.t0) is not None and md.window(ev.t0, max(horizons)) is not None
    ]
    workers = workers if workers is not None else min(len(jobs), os.cpu_count() or 1)
    try:
        if workers > 1:
            if "fork" in mp.get_all_start_methods():
                with ProcessPoolExecutor(max_workers=workers, mp_context=mp.get_context("fork")) as ex:
                    parts = list(ex.map(_replay_one, jobs))
            else:  # no fork (e.g. Windows): run sequentially rather than pickling data per job
                parts = [_replay_one(j) for j in jobs]
        else:
            parts = [_replay_one(j) for j in jobs]
    finally:
        if prev is None:
            os.environ.pop("SENTINEL_DISABLE_LLM", None)
        else:
            os.environ["SENTINEL_DISABLE_LLM"] = prev
    df = pd.DataFrame([row for part in parts for row in part])
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


BOOT_N = 2000
BOOT_LEVEL = 0.90


def bootstrap_skill(err: np.ndarray, base_err: np.ndarray, n: int = BOOT_N, seed: int = 0) -> tuple[float, float]:
    """Percentile interval of 1 - mean(err)/mean(base_err), resampling events with replacement.

    Events are the unit of resampling (the model and the baseline are scored on the same draw),
    so the interval reflects how much the skill depends on which events happen to be in the library.
    """
    err, base_err = np.asarray(err, float), np.asarray(base_err, float)
    ok = np.isfinite(err) & np.isfinite(base_err)
    err, base_err = err[ok], base_err[ok]
    if len(err) < 5:
        return float("nan"), float("nan")
    idx = np.random.default_rng(seed).integers(0, len(err), size=(n, len(err)))
    den = base_err[idx].mean(axis=1)
    skill = 1 - err[idx].mean(axis=1) / np.where(den > 0, den, np.nan)
    a = (1 - BOOT_LEVEL) / 2
    return float(np.nanquantile(skill, a)), float(np.nanquantile(skill, 1 - a))


def summarize(df: pd.DataFrame) -> pd.DataFrame:
    out = []
    for h, g in df.groupby("horizon"):
        err = (g["pred"] - g["realized"]).abs()
        zero_err = g["realized"].abs()
        shock_err = (g["shock_baseline"] - g["realized"]).abs()
        z_lo, z_hi = bootstrap_skill(err.to_numpy(), zero_err.to_numpy())
        s_lo, s_hi = bootstrap_skill(err.to_numpy(), shock_err.to_numpy())
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
                "skill_vs_zero": float(1 - err.mean() / zero_err.mean()) if zero_err.mean() else np.nan,
                "skill_vs_zero_lo": z_lo,
                "skill_vs_zero_hi": z_hi,
                "skill_vs_shock": float(1 - err.mean() / shock_err.mean()) if shock_err.mean() else np.nan,
                "skill_vs_shock_lo": s_lo,
                "skill_vs_shock_hi": s_hi,
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
        f"- 개선율 괄호: 사건 단위 부트스트랩 {BOOT_LEVEL:.0%} 신뢰구간({BOOT_N}회 재표본). 구간이 0을 넘으면 기준선보다 낫다고 말할 수 있음",
        "- 선정 경로: curated(초기 사후 선별), gpr_spike(GPR 급등 규칙), official(공식 발표). 규칙 기반 사건만의 결과가 선택 편향을 뺀 성능에 가까움",
        "",
        "## 요약",
        "| 기간 | 사건 수 | 실측 하락 비율 | 방향 적중 | 방향 적중(충격 기준선) | MAE(통합) | MAE(엔진A) | MAE(엔진B) | MAE(0% 기준선) | MAE(충격 기준선) | 0% 기준선 대비 개선 | 충격 기준선 대비 개선 | 구간 포함률 | 순위상관 | 감지율 |",
        "|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|",
    ]
    ci = lambda r, k: f"{r[k]:+.0%} ({r[k + '_lo']:+.0%} ~ {r[k + '_hi']:+.0%})"  # noqa: E731
    for _, r in summary.iterrows():
        lines.append(
            f"| {r['horizon']} | {r['events']} | {r['share_negative']:.0%} | {r['direction_hit']:.0%} | {r['direction_hit_shock_baseline']:.0%} "
            f"| {r['mae_blend']:.2%} | {r['mae_engine_a']:.2%} | {r['mae_engine_b']:.2%} | {r['mae_zero_baseline']:.2%} "
            f"| {r['mae_shock_baseline']:.2%} | {ci(r, 'skill_vs_zero')} | {ci(r, 'skill_vs_shock')} | {r['iqr_coverage']:.0%} "
            f"| {r['rank_corr']:+.2f} | {r['detected_rate']:.0%} |"
        )
    if "selection" in detail:
        lines += ["", "## 선정 경로별", "| 선정 경로 | 기간 | 사건 수 | MAE(통합) | MAE(0% 기준선) | 0% 기준선 대비 개선 | 충격 기준선 대비 개선 |",
                  "|---|---|---|---|---|---|---|"]
        groups = [("전체", detail), ("규칙 기반(gpr_spike+official)", detail[detail["selection"] != "curated"]),
                  ("초기 선별(curated)", detail[detail["selection"] == "curated"])]
        for label, sub in groups:
            if sub.empty:
                continue
            for _, r in summarize(sub).iterrows():
                lines.append(f"| {label} | {r['horizon']} | {r['events']} | {r['mae_blend']:.2%} | {r['mae_zero_baseline']:.2%} "
                             f"| {ci(r, 'skill_vs_zero')} | {ci(r, 'skill_vs_shock')} |")
    if ablation is not None:
        lines += [
            "",
            "## 비교 실험: 이전 예측 구간 방식 (유사 사건 분위수만 사용)",
            "본 결과는 유사 사건 분산에 GARCH 변동성 예측을 더해 구간을 만듦. 아래는 이전 방식(비교용). 점 예측은 동일.",
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
