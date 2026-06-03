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
