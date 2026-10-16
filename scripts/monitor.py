"""Scheduled monitor (GitHub Actions or any cron).

1. Process every new GDELT 15-minute events file since the last run (code only, no LLM).
2. Once per new US trading session, run the market sentinel on the latest close.
3. Only when something triggers, run the agent (planner/reviewer use the LLM if a key is
   set) and write an Impact Brief to <state-dir>/alerts/.

State lives in <state-dir> (GDELT history, streaks, cooldowns, last processed slot/session),
so a run that is skipped or delayed catches up on the files it missed.

Usage: python scripts/monitor.py [--state-dir monitor_state] [--preset sovereign]
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from sentinel import gdelt_monitor as g  # noqa: E402
from sentinel import monitor as market_monitor  # noqa: E402
from sentinel.agents.pipeline import run  # noqa: E402
from sentinel.data.store import MarketData  # noqa: E402
from sentinel.data.universe import PRESETS  # noqa: E402

MAX_CATCHUP_SLOTS = 16  # four hours of files per run
MAX_ANALYSES = 2  # LLM cost cap per run


def load_state(d: Path) -> tuple[dict, pd.DataFrame]:
    st = json.loads((d / "state.json").read_text()) if (d / "state.json").exists() else {}
    hist_path = d / "gdelt_history.csv"
    hist = pd.read_csv(hist_path, parse_dates=["slot"]) if hist_path.exists() else pd.DataFrame()
    return st, hist


def save_state(d: Path, st: dict, hist: pd.DataFrame) -> None:
    d.mkdir(parents=True, exist_ok=True)
    (d / "state.json").write_text(json.dumps(st, indent=2, ensure_ascii=False))
    hist.to_csv(d / "gdelt_history.csv", index=False)


def gdelt_step(st: dict, hist: pd.DataFrame) -> tuple[list[g.GdeltAlert], pd.DataFrame, str]:
    latest = g.latest_slot()
    last = pd.Timestamp(st["last_slot"]) if st.get("last_slot") else None
    if last is None:  # first run: one day of history, no detection on it
        warm = [latest - i * g.SLOT for i in range(96, 0, -1)]
        data = g.fetch_slots(warm)
        hist = pd.concat([g.aggregate(data[s]) for s in sorted(data)], ignore_index=True) if data else hist
        todo = [latest]
    else:
        n = int((latest - last) / g.SLOT)
        todo = [last + i * g.SLOT for i in range(max(1, n - MAX_CATCHUP_SLOTS + 1), n + 1)] if n > 0 else []
    data = g.fetch_slots(todo)
    alerts: list[g.GdeltAlert] = []
    streak = st.get("streak", {})
    last_alerts = st.get("last_alerts", {})
    for s in sorted(data):
        agg = g.aggregate(data[s])
        found, streak = g.detect(hist, agg, data[s], last_alerts, streak)
        for a in found:
            last_alerts[a.country] = a.slot
        alerts += found
        hist = pd.concat([hist, agg], ignore_index=True)
    st.update(last_slot=str(latest), streak=streak, last_alerts=last_alerts)
    hist = g.trim_history(hist, latest) if len(hist) else hist
    msg = f"GDELT: processed {len(data)}/{len(todo)} new file(s) up to {latest}, alerts {len(alerts)}"
    return alerts, hist, msg


def load_market() -> tuple[MarketData | None, str]:
    try:
        return MarketData.load(offline=False, max_age_hours=6), "online"
    except Exception as exc:  # noqa: BLE001
        try:
            return MarketData.load(offline=True, max_age_hours=None), f"cache (online failed: {type(exc).__name__})"
        except Exception:  # noqa: BLE001
            return None, f"unavailable ({type(exc).__name__})"


def write_alert(d: Path, key: str, title: str, body: str, meta: dict) -> Path:
    out = d / "alerts"
    out.mkdir(parents=True, exist_ok=True)
    stamp = time.strftime("%Y%m%d-%H%M", time.gmtime())
    path = out / f"{stamp}_{key}.md"
    path.write_text(f"# {title}\n\n{body}\n", encoding="utf-8")
    path.with_suffix(".json").write_text(json.dumps(meta, indent=2, ensure_ascii=False, default=str))
    return path


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--state-dir", default="monitor_state")
    ap.add_argument("--preset", default=os.environ.get("SENTINEL_MONITOR_PRESET", "sovereign"))
    args = ap.parse_args()
    d = Path(args.state_dir)
    st, hist = load_state(d)
    log: list[str] = []

    try:
        g_alerts, hist, msg = gdelt_step(st, hist)
    except Exception as exc:  # noqa: BLE001
        g_alerts, msg = [], f"GDELT step failed: {type(exc).__name__}: {exc}"
    log.append(msg)

    md, src = load_market()
    log.append(f"Market data: {src}" + (f", last session {md.last_date.date()}" if md else ""))
    if md is not None and md.problems:
        log.append("Data issues: " + "; ".join(f"{r.source}:{r.key} {r.origin} {r.error}".strip() for r in md.problems))
    market_trigger = None
    if md is not None and str(md.last_date.date()) != st.get("last_market_session"):
        det = market_monitor.detect(md, md.last_date)
        st["last_market_session"] = str(md.last_date.date())
        log.append(f"Market sentinel {md.last_date.date()}: {'TRIGGERED' if det['triggered'] else 'quiet'}")
        if det["triggered"]:
            market_trigger = det

    weights = PRESETS[args.preset].weights
    written: list[Path] = []
    jobs = [("gdelt", a) for a in g_alerts] + ([("market", market_trigger)] if market_trigger else [])
    for kind, item in jobs[:MAX_ANALYSES]:
        if md is None:
            log.append(f"Skip analysis for {kind}: no market data")
            continue
        if kind == "gdelt":
            state = run(md, weights, mode="ask", event_text=item.event_text())
            title = f"[GDELT] {item.name} 분쟁 보도 급증 (z {item.z:.1f})"
            key, meta = f"gdelt_{item.country}", {"trigger": item.as_dict()}
        else:
            state = run(md, weights, mode="monitor")
            title = f"[시장] {item['session']} 이상 변동 감지"
            key, meta = "market", {"trigger": item["top"]}
        meta.update(classification=state.get("classification"), verification=state.get("verification"),
                    facts=state.get("facts"))
        written.append(write_alert(d, key, title, state.get("brief", ""), meta))
    if len(jobs) > MAX_ANALYSES:
        log.append(f"{len(jobs) - MAX_ANALYSES} more trigger(s) not analyzed (cost cap); see state")

    save_state(d, st, hist)
    summary = "\n".join(f"- {line}" for line in log)
    if written:
        bodies = "\n\n---\n\n".join(p.read_text(encoding="utf-8") for p in written)
        (d / "new_alerts.md").write_text(f"{summary}\n\n{bodies}\n", encoding="utf-8")
    elif (d / "new_alerts.md").exists():
        (d / "new_alerts.md").unlink()
    print(summary)
    for p in written:
        print(f"alert written: {p}")
    if os.environ.get("GITHUB_STEP_SUMMARY"):
        with open(os.environ["GITHUB_STEP_SUMMARY"], "a", encoding="utf-8") as f:
            f.write(f"### MACRO SENTINEL monitor\n{summary}\n")
    if os.environ.get("GITHUB_OUTPUT"):
        with open(os.environ["GITHUB_OUTPUT"], "a", encoding="utf-8") as f:
            f.write(f"alerts={len(written)}\n")


if __name__ == "__main__":
    main()
