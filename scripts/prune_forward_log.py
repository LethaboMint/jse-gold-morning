"""Drop predictions_log rows before the configured history window."""
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from audit_forward_log import resolve_log_min_date

LOG = ROOT / "data" / "forward_model" / "predictions_log.csv"


def main() -> int:
    if not LOG.exists():
        return 0
    min_d = resolve_log_min_date()
    if min_d is None:
        return 0
    df = pd.read_csv(LOG)
    df["signal_date"] = pd.to_datetime(df["signal_date"]).dt.normalize()
    before = len(df)
    df = df[df["signal_date"] >= min_d]
    if "run_ts_utc" in df.columns:
        df = df.sort_values("run_ts_utc")
    df = df.drop_duplicates(subset=["signal_date", "miner"], keep="last")
    df["signal_date"] = df["signal_date"].dt.strftime("%Y-%m-%d")
    df.to_csv(LOG, index=False)
    print(f"Pruned log to >= {min_d.date()} ({before} -> {len(df)} rows)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
