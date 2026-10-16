"""Download all series into the local cache (data/cache) and print a per-source report.

Exit code: 0 ok, 1 calendar (SPY) unavailable, 2 some series failed or a quality check errored.

Usage:
  python scripts/build_snapshot.py [--max-age-hours 24]
  python scripts/build_snapshot.py --write-snapshot   # also refresh data/snapshot (public-domain sources only)
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from sentinel.data import cache  # noqa: E402
from sentinel.data.store import DataUnavailable, MarketData  # noqa: E402
from sentinel.data.universe import FACTORS  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--max-age-hours", type=float, default=24)
    ap.add_argument("--write-snapshot", action="store_true")
    args = ap.parse_args()
    try:
        md = MarketData.load(max_age_hours=args.max_age_hours)
        report = md.report
    except DataUnavailable as exc:
        md, report = None, exc.report
    for r in report:
        flag = "OK " if r.origin in ("cache", "network") else "!! "
        print(f"{flag}{r.source:9s} {r.key:16s} {r.origin:12s} {r.error}")
    if md is not None:
        print(f"prices: {md.prices.shape}, {md.calendar[0].date()} ~ {md.last_date.date()}")
        print(f"quality checks: {len(md.quality)} failure(s)")
        for f in md.quality:
            sample = ", ".join(str(d.date()) for d in f.dates)
            print(f"{'!! ' if f.severity == 'error' else '-  '}{f.source:9s} {f.key:16s} {f.check:20s} n={f.n_cases} {sample}")
    if args.write_snapshot:
        for f in FACTORS.values():
            if f.source != "derived" and cache.write_snapshot(f.source, f.source_id):
                print(f"snapshot written: {f.source}/{f.source_id}")
    failed = [r for r in report if r.origin == "failed"]
    q_err = [f for f in md.quality if f.severity == "error"] if md is not None else []
    sys.exit(1 if md is None else 0 if not (failed or q_err) else 2)


if __name__ == "__main__":
    main()
