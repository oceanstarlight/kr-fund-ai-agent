"""Measure how the LLM planner's classification affects results.

For each library event the planner classifies the raw event text (curated tags hidden).
Reports label agreement with the curated tags and how much the D+5 forecast changes
versus the curated-tag run. Compare models by setting SENTINEL_GEMINI_MODEL_FAST (or the
Anthropic equivalents) before running.

Usage: python scripts/eval_planner.py [--preset sovereign] [--out reports/planner_eval.csv]
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from sentinel import config  # noqa: E402
from sentinel.agents import llm  # noqa: E402
from sentinel.agents.pipeline import run  # noqa: E402
from sentinel.data.store import MarketData  # noqa: E402
from sentinel.data.universe import PRESETS  # noqa: E402
from sentinel.events.library import load_events  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--preset", default="sovereign")
    ap.add_argument("--out", default=str(config.REPORT_DIR / "planner_eval.csv"))
    args = ap.parse_args()
    if not llm.llm_available():
        sys.exit("No LLM key configured: nothing to evaluate.")
    md = MarketData.load(offline=True, max_age_hours=None)
    w = PRESETS[args.preset].weights
    h = config.PRIMARY_HORIZON
    rows = []
    for ev in load_events():
        if md.trading_day_before(ev.t0) is None or md.window(ev.t0, max(config.HORIZONS)) is None:
            continue
        os.environ["SENTINEL_DISABLE_LLM"] = "1"
        ref = run(md, w, mode="replay", event_id=ev.id)  # curated tags, rule planner
        del os.environ["SENTINEL_DISABLE_LLM"]
        st = run(md, w, mode="replay", event_id=ev.id, use_curated_tags=False)
        c, t = st["classification"], ev.tags()
        ch_pred, ch_true = set(c["channels"]), set(t["channels"])
        p_ref, p_llm = ref["impacts"][h], st["impacts"][h]
        r = st.get("realized", {}).get(h)
        rows.append({
            "event": ev.id, "planner": st["classifier"],
            "type_ok": c["type"] == t["type"], "region_ok": c["region"] == t["region"],
            "channel_jaccard": len(ch_pred & ch_true) / len(ch_pred | ch_true) if ch_pred | ch_true else 1.0,
            "pred_type": c["type"], "true_type": t["type"],
            "d5_curated": p_ref.blend["q50"], "d5_llm": p_llm.blend["q50"],
            "abs_change": abs(p_llm.blend["q50"] - p_ref.blend["q50"]),
            "realized": r["realized"] if r else float("nan"),
        })
    df = pd.DataFrame(rows)
    df.to_csv(args.out, index=False)
    llm_rows = df[df["planner"].str.startswith("llm")]
    print(f"model: {llm.models()[0]} ({llm.provider()}), events: {len(df)}, LLM-planned: {len(llm_rows)}")
    print(f"type accuracy {df['type_ok'].mean():.0%}, region accuracy {df['region_ok'].mean():.0%}, "
          f"channel Jaccard {df['channel_jaccard'].mean():.2f}")
    print(f"D+5 forecast change vs curated tags: median {df['abs_change'].median():.2%}, max {df['abs_change'].max():.2%}")
    err_cur = (df["d5_curated"] - df["realized"]).abs().mean()
    err_llm = (df["d5_llm"] - df["realized"]).abs().mean()
    print(f"D+5 MAE: curated tags {err_cur:.2%}, LLM classification {err_llm:.2%}")
    print(df[~df["type_ok"]][["event", "pred_type", "true_type"]].to_string(index=False))


if __name__ == "__main__":
    main()
