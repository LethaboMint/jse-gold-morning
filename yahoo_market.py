"""Yahoo Finance market data for gold, GDX, ZAR, and JSE miners."""
from __future__ import annotations

from datetime import datetime, timezone

import numpy as np
import pandas as pd
import yfinance as yf

YF_GOLD = "GC=F"
YF_GDX = "GDX"
YF_ZAR = "ZAR=X"

MINERS = ["HAR", "GFI", "ANG", "DRD", "PAN", "SSW"]
YF_MINERS = {
    "HAR": "HAR.JO",
    "GFI": "GFI.JO",
    "ANG": "AU",
    "DRD": "DRD.JO",
    "PAN": "PAN.JO",
    "SSW": "SSW.JO",
}
# If a JSE ticker fails on Yahoo, try these (ANG.JO is often missing → NYSE AU)
YF_MINER_FALLBACKS: dict[str, list[str]] = {
    "ANG": ["ANG.JO"],
}

HISTORY_START = "2012-05-01"


def _end_date() -> str:
    return str((pd.Timestamp.today() + pd.Timedelta(days=5)).date())


def download_close(ticker: str, start: str = HISTORY_START, end: str | None = None) -> pd.Series:
    import time

    end = end or _end_date()
    last_err: Exception | None = None
    for attempt in range(4):
        raw = yf.download(ticker, start=start, end=end, interval="1d", auto_adjust=False, progress=False)
        if not raw.empty:
            break
        last_err = RuntimeError(f"No Yahoo data for {ticker}")
        time.sleep(1.5 * (attempt + 1))
    else:
        raise last_err or RuntimeError(f"No Yahoo data for {ticker}")
    close = raw["Close"].iloc[:, 0] if isinstance(raw.columns, pd.MultiIndex) else raw["Close"]
    close.index = pd.to_datetime(close.index).normalize()
    return close.sort_index()


def download_miner_close(miner: str, start: str = HISTORY_START, end: str | None = None) -> tuple[pd.Series, str]:
    """Return close series and the Yahoo ticker that worked."""
    primary = YF_MINERS[miner]
    for ticker in [primary, *YF_MINER_FALLBACKS.get(miner, [])]:
        try:
            s = download_close(ticker, start, end)
            if len(s.dropna()) > 252:
                return s, ticker
        except RuntimeError:
            continue
    raise RuntimeError(f"No Yahoo data for miner {miner} ({primary})")


def display_close(close: float | None, ticker: str) -> float | None:
    """Yahoo JSE (*.JO) quotes are usually in ZAR cents."""
    if close is None:
        return None
    if ticker.endswith(".JO") and close > 500:
        return round(close / 100.0, 2)
    return round(close, 4)


def quote_at_date(close: pd.Series, date: pd.Timestamp, ticker: str = "") -> dict:
    """Close and simple daily % change on date (vs prior trading day)."""
    date = pd.Timestamp(date).normalize()
    hist = close.dropna()
    if hist.empty:
        return {"close": None, "pct_change": None, "quote_date": None}

    if date not in hist.index:
        hist = hist.loc[:date]
        if hist.empty:
            return {"close": None, "pct_change": None, "quote_date": None}
        date = hist.index[-1]

    c = float(hist.loc[date])
    pos = hist.index.get_loc(date)
    if isinstance(pos, slice):
        pos = pos.stop - 1
    pct = None
    if pos > 0:
        prev = float(hist.iloc[pos - 1])
        if prev != 0:
            pct = (c / prev - 1.0) * 100.0

    return {
        "close": display_close(c, ticker),
        "pct_change": round(pct, 2) if pct is not None else None,
        "quote_date": str(date.date()),
    }


def build_history_panel() -> tuple[pd.DataFrame, pd.Timestamp]:
    """Panel of log returns for modeling."""
    end = _end_date()
    gold_c = download_close(YF_GOLD, HISTORY_START, end)
    gdx_c = download_close(YF_GDX, HISTORY_START, end)
    zar_c = download_close(YF_ZAR, HISTORY_START, end)

    idx = gold_c.index.union(gdx_c.index).union(zar_c.index)
    miner_closes: dict[str, pd.Series] = {}
    for m in MINERS:
        mc, _ = download_miner_close(m, HISTORY_START, end)
        miner_closes[m] = mc
        idx = idx.union(mc.index)

    panel = pd.DataFrame(
        {
            "return_gold_t": np.log(gold_c).diff().reindex(idx),
            "return_gdx_t": np.log(gdx_c).diff().reindex(idx),
            "return_zar_t": np.log(zar_c).diff().reindex(idx),
        },
        index=idx,
    )

    for m, mc in miner_closes.items():
        panel[f"return_miner_t1_{m}"] = np.log(mc).diff().reindex(idx).shift(-1)

    panel = panel.dropna(subset=["return_gold_t", "return_gdx_t", "return_zar_t"])
    return panel, panel.index.max()


def build_market_snapshot(signal_date: pd.Timestamp) -> dict:
    """Prices and % changes for dashboard (all Yahoo)."""
    end = _end_date()
    gold_c = download_close(YF_GOLD, HISTORY_START, end)
    gdx_c = download_close(YF_GDX, HISTORY_START, end)

    snap = {
        "data_source": "yahoo_finance",
        "fetched_at_utc": datetime.now(timezone.utc).isoformat(),
        "signal_date": str(pd.Timestamp(signal_date).date()),
        "gold": {
            "ticker": YF_GOLD,
            "label": "Gold (COMEX)",
            "currency": "USD",
            **quote_at_date(gold_c, signal_date, YF_GOLD),
        },
        "gdx": {
            "ticker": YF_GDX,
            "label": "GDX",
            "currency": "USD",
            **quote_at_date(gdx_c, signal_date, YF_GDX),
        },
    }

    miners = []
    for m in MINERS:
        mc, ticker = download_miner_close(m, HISTORY_START, end)
        q = quote_at_date(mc, signal_date, ticker)
        ccy = "ZAR" if ticker.endswith(".JO") else "USD"
        miners.append({"miner": m, "ticker": ticker, "label": m, "currency": ccy, **q})
    snap["miners"] = miners
    return snap
