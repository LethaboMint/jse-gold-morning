"""
Backfill predictions_log.csv for past US signal dates (Yahoo panel).

Usage:
  python scripts/backfill_forward_log.py
  python scripts/backfill_forward_log.py --days 60 --rules dashboard
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from score_miners_forward import LOG_PATH, load_history, run_generation


def dedupe_log(path: Path) -> int:
    if not path.exists():
        return 0
    df = pd.read_csv(path)
    before = len(df)
    if "run_ts_utc" in df.columns:
        df = df.sort_values("run_ts_utc")
    df = df.drop_duplicates(subset=["signal_date", "miner"], keep="last")
    df.to_csv(path, index=False)
    return before - len(df)


def main() -> int:
    parser = argparse.ArgumentParser(description="Backfill forward prediction log")
    parser.add_argument("--days", type=int, default=60, help="Trading days to backfill")
    parser.add_argument(
        "--rules",
        choices=("dashboard", "production", "high_conviction", "research", "none"),
        default="dashboard",
    )
    parser.add_argument("--from", dest="from_date", type=str, default=None, help="YYYY-MM-DD earliest signal")
    parser.add_argument("--to", dest="to_date", type=str, default=None, help="YYYY-MM-DD latest signal")
    parser.add_argument("--dedupe-only", action="store_true", help="Only dedupe existing log")
    args = parser.parse_args()

    removed = dedupe_log(LOG_PATH)
    if removed:
        print(f"Deduped log: removed {removed} duplicate rows -> {LOG_PATH}")

    if args.dedupe_only:
        return 0

    panel, last_ts = load_history()
    print("Loaded Yahoo panel once for backfill.")
    dates = panel.index.sort_values()

    if args.to_date:
        dates = dates[dates <= pd.Timestamp(args.to_date).normalize()]
    else:
        dates = dates[dates <= last_ts]

    if args.from_date:
        dates = dates[dates >= pd.Timestamp(args.from_date).normalize()]
    elif args.days > 0:
        dates = dates[-args.days :]

    print(f"Backfilling {len(dates)} signal dates ({dates.min().date()} .. {dates.max().date()}), rules={args.rules}")

    ok = 0
    skipped = 0
    for d in dates:
        try:
            run_generation(
                as_of=str(d.date()),
                rules=args.rules,
                skip_duplicate=True,
                write_latest=False,
                quiet=True,
                with_quotes=False,
                panel=panel,
                panel_last=last_ts,
            )
            ok += 1
        except ValueError:
            skipped += 1
        if ok and ok % 10 == 0:
            print(f"  ... {ok} dates logged")

    removed2 = dedupe_log(LOG_PATH)
    print(f"Done: wrote/kept {ok} dates, skipped {skipped}, deduped {removed2} rows")
    print(f"Log: {LOG_PATH} ({len(pd.read_csv(LOG_PATH))} rows)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
