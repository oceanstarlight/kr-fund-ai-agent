"""Download all series into the local cache (data/cache) and print a summary.

Usage: python scripts/build_snapshot.py [--max-age-hours 24]
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from sentinel.data.store import MarketData  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--max-age-hours", type=float, default=24)
    args = ap.parse_args()
    md = MarketData.load(max_age_hours=args.max_age_hours)
    print(f"prices: {md.prices.shape}, {md.calendar[0].date()} ~ {md.last_date.date()}")
    print(f"factors: {list(md.factors.columns)}")
    print(f"gpr: {None if md.gpr is None else (md.gpr.index[-1].date())}")
    for t, d in sorted(md.inception.items(), key=lambda x: x[1]):
        print(f"  {t:9s} first real data {d.date()}")


if __name__ == "__main__":
    main()
