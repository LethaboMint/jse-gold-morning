"""
Join forward prediction log to realized t+1 miner returns (Yahoo).

Usage:
  python audit_forward_log.py              # score all log rows
  python audit_forward_log.py --summary    # print only summary

Reads:  data/forward_model/predictions_log.csv
Writes: data/forward_model/predictions_audited.csv
        data/forward_model/forward_audit_summary.json
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from yahoo_market import MINERS, download_miner_close

ROOT = Path(__file__).resolve().parent
LOG_PATH = ROOT / "data" / "forward_model" / "predictions_log.csv"
AUDITED_PATH = ROOT / "data" / "forward_model" / "predictions_audited.csv"
SUMMARY_PATH = ROOT / "data" / "forward_model" / "forward_audit_summary.json"


def miner_log_returns() -> pd.DataFrame:
    parts = []
    for m in MINERS:
        c, _ = download_miner_close(m)
        parts.append(np.log(c).diff().rename(m))
    return pd.concat(parts, axis=1).sort_index()


def miner_simple_returns() -> pd.DataFrame:
    """Daily % change from closes (for display)."""
    parts = []
    for m in MINERS:
        c, _ = download_miner_close(m)
        parts.append(c.pct_change().rename(m))
    return pd.concat(parts, axis=1).sort_index()


def return_to_direction(r: float, eps: float = 1e-8) -> str | None:
    if not np.isfinite(r):
        return None
    if abs(r) < eps:
        return "FLAT"
    return "LONG" if r > 0 else "SHORT"


def realized_after(signal_date: pd.Timestamp, series: pd.Series) -> tuple[pd.Timestamp | pd.NaT, float]:
    """First trading row strictly after signal_date."""
    future_idx = series.index[series.index > signal_date]
    if len(future_idx) == 0:
        return pd.NaT, np.nan
    d = future_idx[0]
    return d, float(series.loc[d])


def audit_log(log: pd.DataFrame, ret_log: pd.DataFrame, ret_pct: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for _, r in log.iterrows():
        d = pd.Timestamp(r["signal_date"]).normalize()
        m = r["miner"]
        if m not in ret_log.columns:
            continue

        realized_date, realized_log = realized_after(d, ret_log[m])
        _, realized_pct_s = realized_after(d, ret_pct[m])

        pred = float(r["pred_return_miner_t1"])
        pred_pct = pred * 100.0  # approx for small moves

        pred_dir_forecast = return_to_direction(pred)
        pred_dir = str(r.get("signal", pred_dir_forecast or "")).upper() or pred_dir_forecast
        pred_dir_hiconv = str(r.get("signal_high_conv", "")).upper() or None
        actual_dir = return_to_direction(realized_log) if np.isfinite(realized_log) else None

        if np.isfinite(realized_log):
            hit = float(np.sign(pred) == np.sign(realized_log))
            err_log = realized_log - pred
            err_pct = (realized_pct_s * 100.0 if np.isfinite(realized_pct_s) else np.nan) - pred_pct
            dir_match = (
                float(pred_dir == actual_dir)
                if pred_dir in ("LONG", "SHORT") and actual_dir in ("LONG", "SHORT")
                else np.nan
            )
        else:
            hit = np.nan
            err_log = np.nan
            err_pct = np.nan
            dir_match = np.nan

        if pred_dir_hiconv in ("LONG", "SHORT") and actual_dir in ("LONG", "SHORT"):
            hit_hi = float(pred_dir_hiconv == actual_dir)
        else:
            hit_hi = np.nan

        rows.append(
            {
                **r.to_dict(),
                "realized_date": realized_date.date() if pd.notna(realized_date) else None,
                "predicted_direction": pred_dir,
                "predicted_direction_forecast": pred_dir_forecast,
                "predicted_direction_hiconv": pred_dir_hiconv,
                "actual_direction": actual_dir,
                "direction_match": dir_match,
                "realized_return_log": realized_log,
                "realized_return_pct": realized_pct_s * 100.0 if np.isfinite(realized_pct_s) else np.nan,
                "pred_return_pct": pred_pct,
                "error_log": err_log,
                "error_pct": err_pct,
                "hit_direction": hit,
                "hit_high_conv": hit_hi,
            }
        )
    return pd.DataFrame(rows)


def build_summary(scored: pd.DataFrame) -> dict:
    overall = {
        "n": int(len(scored)),
        "hit_rate": float(scored["hit_direction"].mean()) if len(scored) else None,
    }
    by_miner = (
        scored.groupby("miner")
        .agg(
            n=("hit_direction", "count"),
            hit_rate=("hit_direction", "mean"),
            mae_pct=("error_pct", lambda s: float(np.nanmean(np.abs(s)))),
            mean_pred_pct=("pred_return_pct", "mean"),
            mean_real_pct=("realized_return_pct", "mean"),
        )
        .round(4)
        .reset_index()
        .to_dict(orient="records")
    )

    hi = scored[scored["signal_high_conv"].isin(["LONG", "SHORT"])]
    hi_summary = None
    if len(hi) > 0:
        hi_summary = {
            "n": int(len(hi)),
            "hit_rate": float(hi["hit_high_conv"].mean()),
        }

    recent = scored.groupby("signal_date", as_index=False).agg(
        n=("hit_direction", "count"), hit_rate=("hit_direction", "mean")
    )
    recent["signal_date"] = recent["signal_date"].astype(str)
    by_signal_date = recent.tail(20).round(4).to_dict(orient="records")

    return {
        "overall": overall,
        "high_conviction": hi_summary,
        "by_miner": by_miner,
        "recent_by_signal_date": by_signal_date,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Audit predictions vs realized t+1")
    parser.add_argument("--summary", action="store_true", help="Print summary only")
    args = parser.parse_args()

    if not LOG_PATH.exists():
        print(f"No log yet: {LOG_PATH}")
        print("Run daily: python generate_daily_signals.py")
        return

    log = pd.read_csv(LOG_PATH)
    log["signal_date"] = pd.to_datetime(log["signal_date"]).dt.normalize()

    print("Fetching Yahoo miner history for realized returns...")
    ret_log = miner_log_returns()
    ret_pct = miner_simple_returns()

    out = audit_log(log, ret_log, ret_pct)
    out.to_csv(AUDITED_PATH, index=False)

    scored = out.dropna(subset=["realized_return_log"])
    summary = build_summary(scored)
    summary["audited_at_utc"] = pd.Timestamp.utcnow().isoformat()
    summary["log_rows"] = int(len(log))
    summary["scored_rows"] = int(len(scored))
    SUMMARY_PATH.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    # Publish recent rows for the website
    audit_doc = ROOT / "docs" / "audit.json"
    recent_cols = [
        "signal_date",
        "realized_date",
        "miner",
        "pred_return_pct",
        "realized_return_pct",
        "predicted_direction",
        "actual_direction",
        "predicted_direction_hiconv",
        "direction_match",
        "hit_high_conv",
    ]
    recent = out.dropna(subset=["realized_return_log"]).tail(60)
    payload = {
        "audited_at_utc": summary["audited_at_utc"],
        "overall_hit_rate": summary["overall"].get("hit_rate"),
        "rows": recent[[c for c in recent_cols if c in recent.columns]].to_dict(orient="records"),
    }
    for row in payload["rows"]:
        for k in ("signal_date", "realized_date"):
            if k in row and row[k] is not None:
                row[k] = str(row[k])[:10]
    audit_doc.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    if args.summary and len(scored) == 0:
        print("No scored rows yet — wait until t+1 JSE data exists for logged signal dates.")
        return

    print(f"Log rows: {len(log)} | Scored (have t+1 realized): {len(scored)}")
    if len(scored) == 0:
        print("Nothing to score yet. Each signal_date needs a later JSE trading day in Yahoo data.")
        print(f"Saved empty audit: {AUDITED_PATH}")
        return

    print(f"\nOverall direction hit rate: {scored['hit_direction'].mean():.1%} (n={len(scored)})")
    if summary.get("high_conviction"):
        hc = summary["high_conviction"]
        print(f"High-conviction hit rate: {hc['hit_rate']:.1%} (n={hc['n']})")

    print("\nPer-miner:")
    print(
        scored.groupby("miner")
        .agg(
            n=("hit_direction", "count"),
            hit=("hit_direction", "mean"),
            mae_pct=("error_pct", lambda s: np.nanmean(np.abs(s))),
        )
        .round(3)
        .to_string()
    )

    print(f"\nSaved: {AUDITED_PATH}")
    print(f"Saved: {SUMMARY_PATH}")
    print(f"Saved: {ROOT / 'docs' / 'audit.json'}")


if __name__ == "__main__":
    main()
