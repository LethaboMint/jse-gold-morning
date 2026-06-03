"""Shared regime masks for OLS direction rules."""
from __future__ import annotations

import numpy as np
import pandas as pd

REGIMES = {
    "any": lambda d, p: np.ones(len(d), bool),
    "gold_up": lambda d, p: d["r_gold"].values > p["kg"],
    "gold_down": lambda d, p: d["r_gold"].values < -p["kg"],
    "gdx_up": lambda d, p: d["r_gdx"].values > p["kx"],
    "gdx_down": lambda d, p: d["r_gdx"].values < -p["kx"],
    "gold_up_gdx_up": lambda d, p: (d["r_gold"].values > p["kg"]) & (d["r_gdx"].values > p["kx"]),
    "gold_down_gdx_down": lambda d, p: (d["r_gold"].values < -p["kg"]) & (d["r_gdx"].values < -p["kx"]),
    "gold_up_gdx_down": lambda d, p: (d["r_gold"].values > p["kg"]) & (d["r_gdx"].values < -p["kx"]),
    "gold_down_gdx_up": lambda d, p: (d["r_gold"].values < -p["kg"]) & (d["r_gdx"].values > p["kx"]),
    "zar_weak": lambda d, p: d["r_zar"].values > p["kz"],
    "zar_strong": lambda d, p: d["r_zar"].values < -p["kz"],
}


def regime_mask(
    r_gold: float,
    r_gdx: float,
    r_zar: float,
    regime: str,
    kg: float,
    kx: float,
    kz: float,
) -> bool:
    row = pd.DataFrame({"r_gold": [r_gold], "r_gdx": [r_gdx], "r_zar": [r_zar]})
    fn = REGIMES.get(regime, REGIMES["any"])
    return bool(fn(row, {"kg": kg, "kx": kx, "kz": kz})[0])


def explain_filter(
    pred: float,
    r_gold: float,
    r_gdx: float,
    r_zar: float,
    regime: str,
    kg: float,
    kx: float,
    kz: float,
    pmin: float,
) -> tuple[bool, str]:
    """Return (passes, short reason for dashboard)."""
    pct = lambda x: f"{x * 100:+.2f}%"
    if not regime_mask(r_gold, r_gdx, r_zar, regime, kg, kx, kz):
        if regime == "gold_up":
            return False, f"Gold {pct(r_gold)} (need >{kg*100:.1f}%)"
        if regime == "gold_down":
            return False, f"Gold {pct(r_gold)} (need <−{kg*100:.1f}%)"
        if regime == "gdx_up":
            return False, f"GDX {pct(r_gdx)} (need >{kx*100:.1f}%)"
        if regime == "gdx_down":
            return False, f"GDX {pct(r_gdx)} (need <−{kx*100:.1f}%)"
        if regime == "gold_up_gdx_up":
            return False, f"Gold {pct(r_gold)} & GDX {pct(r_gdx)} (need both up)"
        if regime == "gold_down_gdx_up":
            return False, f"Gold {pct(r_gold)} down & GDX {pct(r_gdx)} up required"
        if regime == "zar_weak":
            return False, f"ZAR {pct(r_zar)} (need weak >{kz*100:.2f}%)"
        if regime == "zar_strong":
            return False, f"ZAR {pct(r_zar)} (need strong <−{kz*100:.2f}%)"
        return False, f"Regime '{regime}' not met"
    if abs(pred) < pmin:
        return False, f"|forecast| {abs(pred)*100:.2f}% < {pmin*100:.1f}%"
    return True, "Filters OK"
