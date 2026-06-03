"""
Forward scorer: fit gold + GDX -> next-day miner return, then signal for latest close.

Usage:
  python score_miners_forward.py              # fit on all history, score latest day
  python score_miners_forward.py --as-of 2026-05-27   # score as if that was "today"

Outputs:
  data/forward_model/coefficients.json
  data/forward_model/predictions_log.csv  (appends one row per run)
"""
from __future__ import annotations

import argparse
import json
import os
from datetime import datetime, timezone
from pathlib import Path

import duckdb
import numpy as np
import pandas as pd
import yfinance as yf

from regime_filters import regime_mask

ROOT = Path(__file__).resolve().parent
DUCKDB_PATH = ROOT / "data" / "market.duckdb"
OUT_DIR = ROOT / "data" / "forward_model"
OUT_DIR.mkdir(parents=True, exist_ok=True)

COEF_PATH = OUT_DIR / "coefficients.json"
LOG_PATH = OUT_DIR / "predictions_log.csv"
LATEST_JSON = OUT_DIR / "latest_signals.json"
LATEST_CSV = OUT_DIR / "latest_signals.csv"

GOLD_SYMBOL = "XAUUSD"
YF_GOLD = "GC=F"
YF_GDX = "GDX"
YF_ZAR = "ZAR=X"
MINERS = ["HAR", "GFI", "ANG", "DRD", "PAN", "SSW"]
YF_MINERS = {
    "HAR": "HAR.JO",
    "GFI": "GFI.JO",
    "ANG": "ANG.JO",
    "DRD": "DRD.JO",
    "PAN": "PAN.JO",
    "SSW": "SSW.JO",
}
PROD_RULES_PATH = ROOT / "data" / "production_rules.json"
HIGH_CONV_PATH = ROOT / "data" / "high_conviction_rules.json"
RESEARCH_PATH = ROOT / "data" / "research_best_holdout_rules.json"

LOG_COLS = [
    "run_ts_utc",
    "signal_date",
    "target_date_note",
    "miner",
    "return_gold_t",
    "return_gdx_t",
    "return_zar_t",
    "pred_return_miner_t1",
    "signal",
    "signal_high_conv",
    "regime_pass",
    "pred_abs_gte",
    "alpha",
    "beta_gold",
    "beta_gdx",
    "n_train",
    "train_end",
]


def ols_fit(X: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Returns [alpha, beta_1, ..., beta_k]."""
    Xc = np.column_stack([np.ones(len(X)), X])
    beta, _, _, _ = np.linalg.lstsq(Xc, y, rcond=None)
    return beta


def download_log_return(ticker: str, start: str, end: str) -> pd.Series:
    raw = yf.download(ticker, start=start, end=end, interval="1d", auto_adjust=False, progress=False)
    if raw.empty:
        raise RuntimeError(f"No Yahoo data for {ticker}")
    close = raw["Close"].iloc[:, 0] if isinstance(raw.columns, pd.MultiIndex) else raw["Close"]
    s = np.log(close).diff()
    s.index = pd.to_datetime(s.index).normalize()
    return s


def load_history_yahoo() -> tuple[pd.DataFrame, pd.Timestamp]:
    """Load panel from Yahoo only (GitHub Actions / no DuckDB)."""
    start = "2012-05-01"
    end = str((pd.Timestamp.today() + pd.Timedelta(days=5)).date())
    yf_gold = download_log_return(YF_GOLD, start, end)
    yf_gdx = download_log_return(YF_GDX, start, end)
    yf_zar = download_log_return(YF_ZAR, start, end)

    miner_ret = {}
    for m, ticker in YF_MINERS.items():
        miner_ret[m] = download_log_return(ticker, start, end)

    idx = yf_gold.index.union(yf_gdx.index).union(yf_zar.index)
    for s in miner_ret.values():
        idx = idx.union(s.index)
    idx = idx.sort_values()

    panel = pd.DataFrame(
        {
            "return_gold_t": yf_gold.reindex(idx),
            "return_gdx_t": yf_gdx.reindex(idx),
            "return_zar_t": yf_zar.reindex(idx),
        },
        index=idx,
    )
    for m, s in miner_ret.items():
        panel[f"return_miner_t1_{m}"] = s.reindex(idx).shift(-1)

    panel = panel.dropna(subset=["return_gold_t", "return_gdx_t", "return_zar_t"])
    return panel, panel.index.max()


def load_history() -> tuple[pd.DataFrame, pd.Timestamp]:
    use_yahoo = os.environ.get("USE_YAHOO_ONLY", "").lower() in ("1", "true", "yes")
    if use_yahoo or not DUCKDB_PATH.exists():
        return load_history_yahoo()

    con = duckdb.connect(str(DUCKDB_PATH), read_only=True)
    syms = ",".join(f"'{s}'" for s in [GOLD_SYMBOL] + MINERS)
    px = con.execute(
        f"SELECT symbol, ts, close FROM ohlcv WHERE symbol IN ({syms}) ORDER BY symbol, ts"
    ).df()
    con.close()

    wide = px.pivot(index="ts", columns="symbol", values="close").sort_index()
    wide.index = pd.to_datetime(wide.index).normalize()
    ret = np.log(wide).diff()

    start = str(wide.index.min().date())
    end = str((wide.index.max() + pd.Timedelta(days=5)).date())

    # Refresh gold/GDX from Yahoo through latest (DuckDB gold may lag)
    yf_gold = download_log_return(YF_GOLD, start, end).rename("return_gold_t")
    yf_gdx = download_log_return(YF_GDX, start, end).rename("return_gdx_t")
    yf_zar = download_log_return(YF_ZAR, start, end).rename("return_zar_t")

    gold_db = ret[GOLD_SYMBOL].rename("return_gold_t_db")
    panel = pd.DataFrame({"return_gold_t": yf_gold.combine_first(gold_db)})

    for m in MINERS:
        panel[f"return_miner_t1_{m}"] = ret[m].shift(-1)

    panel["return_gdx_t"] = yf_gdx
    panel["return_zar_t"] = yf_zar
    panel = panel.dropna(subset=["return_gold_t", "return_gdx_t", "return_zar_t"])
    last_ts = panel.index.max()
    return panel, last_ts


def fit_miner_models(panel: pd.DataFrame, train_end: pd.Timestamp) -> dict:
    train = panel[panel.index <= train_end].copy()
    coeffs = {}
    xcols = ["return_gold_t", "return_gdx_t"]

    for m in MINERS:
        ycol = f"return_miner_t1_{m}"
        d = train[[ycol] + xcols].dropna()
        if len(d) < 252:
            coeffs[m] = None
            continue
        y = d[ycol].values
        X = d[xcols].values
        b = ols_fit(X, y)
        coeffs[m] = {
            "alpha": float(b[0]),
            "beta_gold": float(b[1]),
            "beta_gdx": float(b[2]),
            "n_train": int(len(d)),
            "train_end": str(train_end.date()),
        }
    return coeffs


def predict_one(
    coeffs: dict,
    return_gold_t: float,
    return_gdx_t: float,
) -> float:
    return coeffs["alpha"] + coeffs["beta_gold"] * return_gold_t + coeffs["beta_gdx"] * return_gdx_t


def load_rules(path: Path) -> dict | None:
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def direction_signal(pred: float, min_pred: float) -> str:
    if pred > min_pred:
        return "LONG"
    if pred < -min_pred:
        return "SHORT"
    return "FLAT"


def high_conv_signal(
    pred: float,
    r_gold: float,
    r_gdx: float,
    r_zar: float,
    rule: dict,
) -> tuple[str, bool]:
    """Production rule: regime filter + |pred| threshold; else FLAT."""
    flt = rule.get("filter", {})
    regime = rule.get("regime", "any")
    kg = float(flt.get("kg", flt.get("gold_gt", 0.0)) or 0.0)
    kx = float(flt.get("kx", flt.get("gdx_gt", 0.0)) or 0.0)
    kz = float(flt.get("kz", 0.0) or 0.0)
    pmin = float(flt.get("pred_abs_gte", 0.0) or 0.0)
    if not regime_mask(r_gold, r_gdx, r_zar, regime, kg, kx, kz):
        return "FLAT", False
    if abs(pred) < pmin:
        return "FLAT", False
    return direction_signal(pred, 0.0), True


def append_log(rows: list[dict], skip_duplicate: bool = False) -> bool:
    """Append rows to predictions_log. Returns False if skipped as duplicate."""
    if not rows:
        return True
    new = pd.DataFrame(rows)
    if skip_duplicate and LOG_PATH.exists():
        old = pd.read_csv(LOG_PATH)
        key_date = new["signal_date"].iloc[0]
        key_rules = new.get("rules_mode", pd.Series([""])).iloc[0]
        dup = old[(old["signal_date"] == key_date) & (old.get("rules_mode", "") == key_rules)]
        if len(dup) >= len(MINERS):
            return False
    if LOG_PATH.exists():
        old = pd.read_csv(LOG_PATH)
        out = pd.concat([old, new], ignore_index=True)
    else:
        out = new
    out.to_csv(LOG_PATH, index=False)
    return True


def write_latest_snapshot(rows: list[dict], meta: dict) -> None:
    payload = {
        "generated_at_utc": meta["fitted_at_utc"],
        "signal_date": meta["signal_date"],
        "rules_mode": meta.get("rules_mode"),
        "return_gold_t": meta.get("return_gold_t"),
        "return_gdx_t": meta.get("return_gdx_t"),
        "return_zar_t": meta.get("return_zar_t"),
        "signals": rows,
    }
    LATEST_JSON.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    pd.DataFrame(rows).to_csv(LATEST_CSV, index=False)


def run_generation(
    *,
    as_of: str | None = None,
    min_pred: float = 0.0,
    rules: str = "production",
    skip_duplicate: bool = False,
    write_latest: bool = False,
    quiet: bool = False,
) -> pd.DataFrame:
    rules_path = {
        "production": PROD_RULES_PATH,
        "high_conviction": HIGH_CONV_PATH,
        "research": RESEARCH_PATH,
        "none": None,
    }[rules]

    panel, last_ts = load_history()

    if as_of:
        signal_date = pd.Timestamp(as_of).normalize()
        if signal_date not in panel.index:
            raise ValueError(f"{signal_date.date()} not in panel. Last date: {last_ts.date()}")
    else:
        signal_date = last_ts

    train_end = panel.index[panel.index < signal_date].max() if signal_date != panel.index.min() else signal_date
    if pd.isna(train_end):
        train_end = signal_date

    fit_panel = panel[panel.index <= train_end]
    coeffs_all = fit_miner_models(fit_panel, train_end)

    row = panel.loc[signal_date]
    r_gold = float(row["return_gold_t"])
    r_gdx = float(row["return_gdx_t"])
    r_zar = float(row["return_zar_t"])
    prod = load_rules(rules_path) if rules_path else None

    meta = {
        "fitted_at_utc": datetime.now(timezone.utc).isoformat(),
        "signal_date": str(signal_date.date()),
        "rules_mode": rules,
        "return_gold_t": r_gold,
        "return_gdx_t": r_gdx,
        "return_zar_t": r_zar,
        "interpretation": "Features are log returns on signal_date; prediction is for the next trading session.",
        "miners": coeffs_all,
    }
    COEF_PATH.write_text(json.dumps(meta, indent=2), encoding="utf-8")

    run_ts = datetime.now(timezone.utc).isoformat()
    log_rows = []
    if not quiet:
        print(f"Signal date (features): {signal_date.date()}")
        print(f"Train through:          {train_end.date()}")
        print(f"Gold return (t):        {r_gold*100:+.3f}%")
        print(f"GDX return (t):         {r_gdx*100:+.3f}%")
        print(f"ZAR return (t):         {r_zar*100:+.3f}%")
        print(f"Rules:                  {rules}")
        print()
        print(f"{'Miner':<6} {'Pred t+1':>12} {'Base':>8} {'HiConv':>8}  regime")
        print("-" * 72)

    for m in MINERS:
        c = coeffs_all.get(m)
        if c is None:
            if not quiet:
                print(f"{m:<6} {'(no fit)':>12}")
            continue
        pred = predict_one(c, r_gold, r_gdx)
        sig = direction_signal(pred, min_pred)
        miner_rule = (prod or {}).get("miners", {}).get(m) if prod else None
        if miner_rule:
            sig_hi, regime_ok = high_conv_signal(pred, r_gold, r_gdx, r_zar, miner_rule)
            regime_lbl = miner_rule.get("regime", "?") if regime_ok else "-"
        else:
            sig_hi, regime_lbl = sig, "n/a"
        if not quiet:
            print(f"{m:<6} {pred*100:+11.4f}% {sig:>8} {sig_hi:>8}  {regime_lbl}")
        flt = (miner_rule or {}).get("filter", {})
        log_rows.append(
            {
                "run_ts_utc": run_ts,
                "signal_date": str(signal_date.date()),
                "rules_mode": rules,
                "target_date_note": "next_trading_day_after_signal_date",
                "miner": m,
                "return_gold_t": r_gold,
                "return_gdx_t": r_gdx,
                "return_zar_t": r_zar,
                "pred_return_miner_t1": pred,
                "signal": sig,
                "signal_high_conv": sig_hi,
                "regime_pass": regime_lbl,
                "pred_abs_gte": flt.get("pred_abs_gte"),
                "alpha": c["alpha"],
                "beta_gold": c["beta_gold"],
                "beta_gdx": c["beta_gdx"],
                "n_train": c["n_train"],
                "train_end": c["train_end"],
            }
        )

    wrote = append_log(log_rows, skip_duplicate=skip_duplicate)
    if write_latest:
        write_latest_snapshot(log_rows, meta)
    if not quiet:
        if wrote:
            print(f"\nAppended to log: {LOG_PATH}")
        else:
            print(f"\nSkipped log append (duplicate for {signal_date.date()} / {rules})")
        if write_latest:
            print(f"Latest snapshot:  {LATEST_JSON}")
    return pd.DataFrame(log_rows)


def main() -> None:
    parser = argparse.ArgumentParser(description="Forward score: gold+GDX -> next-day miner direction")
    parser.add_argument(
        "--as-of",
        type=str,
        default=None,
        help="Signal date YYYY-MM-DD (default: latest date in data)",
    )
    parser.add_argument(
        "--min-pred",
        type=float,
        default=0.0,
        help="Only call LONG if pred > min-pred; SHORT if pred < -min-pred else FLAT",
    )
    parser.add_argument(
        "--rules",
        choices=("production", "high_conviction", "research", "none"),
        default="production",
        help="Regime filters: production (WF, default for deploy), research, high_conviction, none",
    )
    parser.add_argument(
        "--write-latest",
        action="store_true",
        help="Write data/forward_model/latest_signals.json and .csv",
    )
    parser.add_argument(
        "--skip-duplicate",
        action="store_true",
        help="Skip append if this signal_date + rules_mode already logged",
    )
    args = parser.parse_args()
    run_generation(
        as_of=args.as_of,
        min_pred=args.min_pred,
        rules=args.rules,
        skip_duplicate=args.skip_duplicate,
        write_latest=args.write_latest,
    )
    print("\nNote: Run once per day after gold/GDX closes. Audit with: python audit_forward_log.py")


if __name__ == "__main__":
    main()
