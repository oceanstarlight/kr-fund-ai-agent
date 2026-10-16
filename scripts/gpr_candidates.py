"""List candidate geopolitical events for the library by a fixed, market-blind rule.

Rule (fixed before looking at any market reaction):
  - daily GPR (GPRD), log scale; z = (log GPRD - rolling median of the prior 90 days)
    / rolling std of the prior 90 days
  - keep days with z >= --z (default 3.0) since --start (default 2000-01-01)
  - greedy by z, at least --gap days (default 30) apart; spikes within --near days of a
    library event are reported as already covered
The script only lists dates; the event behind each spike is identified from news archives
and recorded in events.json with "selection": "gpr_spike".

Usage: python scripts/gpr_candidates.py [--z 3.0] [--start 2000-01-01]
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from sentinel.data import cache  # noqa: E402
from sentinel.events.library import load_events  # noqa: E402


def candidates(gpr: pd.Series, z_min: float, start: str, gap: int, near: int) -> pd.DataFrame:
    lg = np.log(gpr.clip(lower=1e-6))
    z = (lg - lg.rolling(90).median().shift(1)) / lg.rolling(90).std().shift(1)
    z = z[(z.index >= start) & (z >= z_min)].sort_values(ascending=False)
    lib = [pd.Timestamp(e.date) for e in load_events()]
    picked = []
    for d, v in z.items():
        if any(abs((d - p).days) < gap for p, _ in picked):
            continue
        picked.append((d, v))
    rows = [{"date": d.date(), "z": round(float(v), 2), "gprd": round(float(gpr[d]), 0),
             "in_library": any(abs((d - e).days) <= near for e in lib)} for d, v in picked]
    return pd.DataFrame(rows).sort_values("date").reset_index(drop=True)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--z", type=float, default=3.0)
    ap.add_argument("--start", default="2000-01-01")
    ap.add_argument("--gap", type=int, default=30)
    ap.add_argument("--near", type=int, default=10)
    args = ap.parse_args()
    gpr, status = cache.get_gpr(max_age_hours=None, offline=True)
    if gpr is None:
        gpr, status = cache.get_gpr()
    if gpr is None:
        sys.exit(f"GPR unavailable: {status.error}")
    print(candidates(gpr["GPRD"].dropna(), args.z, args.start, args.gap, args.near).to_string(index=False))


if __name__ == "__main__":
    main()
