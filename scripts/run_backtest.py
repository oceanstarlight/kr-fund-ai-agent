"""Run the leave-one-out backtest for every preset and write reports/backtest_<preset>.{csv,md}.

Usage: python scripts/run_backtest.py [--offline]
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from sentinel import config  # noqa: E402
from sentinel.backtest import run_backtest, summarize, to_markdown  # noqa: E402
from sentinel.data.store import MarketData  # noqa: E402
from sentinel.data.universe import PRESETS  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--offline", action="store_true", help="use cached data only")
    args = ap.parse_args()
    md = MarketData.load(offline=args.offline, max_age_hours=None if args.offline else 24)
    config.REPORT_DIR.mkdir(exist_ok=True)
    for key, preset in PRESETS.items():
        detail = run_backtest(md, preset.weights)
        summary = summarize(detail)
        ablation = summarize(run_backtest(md, preset.weights, vol_scale=False))
        detail.to_csv(config.REPORT_DIR / f"backtest_{key}.csv", index=False)
        (config.REPORT_DIR / f"backtest_{key}.md").write_text(
            to_markdown(summary, detail, preset.name, ablation), encoding="utf-8"
        )
        print(f"== {preset.name}")
        print(summary.to_string(index=False, float_format=lambda x: f"{x:.3f}"))


if __name__ == "__main__":
    main()
