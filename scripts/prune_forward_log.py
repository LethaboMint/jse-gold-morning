"""Drop predictions_log rows before signal_config.log_min_signal_date."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
LOG = ROOT / "data" / "forward_model" / "predictions_log.csv"
CONFIG = ROOT / "signal_config.json"


def main() -> int:
    if not CONFIG.exists() or not LOG.exists():
        return 0
    min_raw = json.loads(CONFIG.read_text(encoding="utf-8")).get("log_min_signal_date")
    if not min_raw:
        return 0
    min_d = pd.Timestamp(min_raw).normalize()
    df = pd.read_csv(LOG)
    df["signal_date"] = pd.to_datetime(df["signal_date"]).dt.normalize()
    before = len(df)
    df = df[df["signal_date"] >= min_d]
    if "run_ts_utc" in df.columns:
        df = df.sort_values("run_ts_utc")
    df = df.drop_duplicates(subset=["signal_date", "miner"], keep="last")
    df["signal_date"] = df["signal_date"].dt.strftime("%Y-%m-%d")
    df.to_csv(LOG, index=False)
    print(f"Pruned log to >= {min_d.date()}: {before} -> {len(df)} rows")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
