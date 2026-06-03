"""
Forward scorer: Yahoo gold + GDX -> next-day JSE miner direction.

Usage:
  python score_miners_forward.py
  python generate_daily_signals.py   # scheduled deploy entry point
"""
from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from regime_filters import regime_mask
from yahoo_market import MINERS, YF_MINERS, build_history_panel, build_market_snapshot

ROOT = Path(__file__).resolve().parent
OUT_DIR = ROOT / "data" / "forward_model"
OUT_DIR.mkdir(parents=True, exist_ok=True)

COEF_PATH = OUT_DIR / "coefficients.json"
LOG_PATH = OUT_DIR / "predictions_log.csv"
LATEST_JSON = OUT_DIR / "latest_signals.json"
LATEST_CSV = OUT_DIR / "latest_signals.csv"

PROD_RULES_PATH = ROOT / "data" / "production_rules.json"
HIGH_CONV_PATH = ROOT / "data" / "high_conviction_rules.json"
RESEARCH_PATH = ROOT / "data" / "research_best_holdout_rules.json"


def ols_fit(X: np.ndarray, y: np.ndarray) -> np.ndarray:
    Xc = np.column_stack([np.ones(len(X)), X])
    beta, _, _, _ = np.linalg.lstsq(Xc, y, rcond=None)
    return beta


def load_history() -> tuple[pd.DataFrame, pd.Timestamp]:
    return build_history_panel()


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


def predict_one(coeffs: dict, return_gold_t: float, return_gdx_t: float) -> float:
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


def write_latest_snapshot(rows: list[dict], meta: dict, market: dict) -> None:
    payload = {
        "generated_at_utc": meta["fitted_at_utc"],
        "signal_date": meta["signal_date"],
        "rules_mode": meta.get("rules_mode"),
        "data_source": "yahoo_finance",
        "market": market,
        "return_gold_t": meta.get("return_gold_t"),
        "return_gdx_t": meta.get("return_gdx_t"),
        "return_zar_t": meta.get("return_zar_t"),
        "signals": rows,
    }
    LATEST_JSON.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    pd.DataFrame(rows).to_csv(LATEST_CSV, index=False)


def _miner_quotes_by_symbol(market: dict) -> dict[str, dict]:
    return {m["miner"]: m for m in market.get("miners", [])}


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

    market = build_market_snapshot(signal_date)
    fit_panel = panel[panel.index <= train_end]
    coeffs_all = fit_miner_models(fit_panel, train_end)

    row = panel.loc[signal_date]
    r_gold = float(row["return_gold_t"])
    r_gdx = float(row["return_gdx_t"])
    r_zar = float(row["return_zar_t"])
    prod = load_rules(rules_path) if rules_path else None
    quotes = _miner_quotes_by_symbol(market)

    meta = {
        "fitted_at_utc": datetime.now(timezone.utc).isoformat(),
        "signal_date": str(signal_date.date()),
        "rules_mode": rules,
        "return_gold_t": r_gold,
        "return_gdx_t": r_gdx,
        "return_zar_t": r_zar,
        "miners": coeffs_all,
    }
    COEF_PATH.write_text(json.dumps(meta, indent=2), encoding="utf-8")

    run_ts = datetime.now(timezone.utc).isoformat()
    log_rows = []

    if not quiet:
        g = market["gold"]
        x = market["gdx"]
        print(f"Signal date (US features): {signal_date.date()}")
        print(f"Train through:             {train_end.date()}")
        print(f"Data:                      Yahoo Finance")
        print()
        print(f"Gold {g['ticker']}:  {g['close']} USD  ({g['pct_change']:+.2f}%)" if g.get("pct_change") is not None else f"Gold: {g.get('close')}")
        print(f"GDX  {x['ticker']}:  {x['close']} USD  ({x['pct_change']:+.2f}%)" if x.get("pct_change") is not None else f"GDX: {x.get('close')}")
        print()
        print(f"{'Miner':<6} {'Price':>10} {'Chg%':>8} {'Pred t+1':>10} {'Base':>7} {'HiConv':>7}")
        print("-" * 58)

    for m in MINERS:
        c = coeffs_all.get(m)
        q = quotes.get(m, {})
        price = q.get("close")
        chg = q.get("pct_change")
        ticker = q.get("ticker", YF_MINERS.get(m, m))

        if c is None:
            if not quiet:
                print(f"{m:<6} {'—':>10} {'—':>8} {'(no fit)':>10}")
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
            ps = f"{price:,.2f}" if price is not None else "—"
            cs = f"{chg:+.2f}%" if chg is not None else "—"
            print(f"{m:<6} {ps:>10} {cs:>8} {pred*100:+9.3f}% {sig:>7} {sig_hi:>7}")

        flt = (miner_rule or {}).get("filter", {})
        log_rows.append(
            {
                "run_ts_utc": run_ts,
                "signal_date": str(signal_date.date()),
                "rules_mode": rules,
                "miner": m,
                "yahoo_ticker": ticker,
                "close": price,
                "pct_change": chg,
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
        write_latest_snapshot(log_rows, meta, market)
    if not quiet:
        if wrote:
            print(f"\nLog: {LOG_PATH}")
        else:
            print(f"\nSkipped duplicate log for {signal_date.date()}")
        if write_latest:
            print(f"Site data: {LATEST_JSON}")
    return pd.DataFrame(log_rows)


def main() -> None:
    parser = argparse.ArgumentParser(description="Yahoo gold+GDX -> next-day miner signals")
    parser.add_argument("--as-of", type=str, default=None)
    parser.add_argument("--min-pred", type=float, default=0.0)
    parser.add_argument(
        "--rules",
        choices=("production", "high_conviction", "research", "none"),
        default="production",
    )
    parser.add_argument("--write-latest", action="store_true")
    parser.add_argument("--skip-duplicate", action="store_true")
    args = parser.parse_args()
    run_generation(
        as_of=args.as_of,
        min_pred=args.min_pred,
        rules=args.rules,
        skip_duplicate=args.skip_duplicate,
        write_latest=args.write_latest,
    )


if __name__ == "__main__":
    main()
