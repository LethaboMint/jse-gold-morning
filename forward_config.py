"""Shared forward-hold horizon for model, audit, and dashboard."""
from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent
CONFIG_PATH = ROOT / "signal_config.json"


def load_config() -> dict:
    if not CONFIG_PATH.exists():
        return {}
    return json.loads(CONFIG_PATH.read_text(encoding="utf-8"))


def forward_horizon_days() -> int:
    """Trading sessions to hold / model forward return (default ~1 month)."""
    cfg = load_config()
    return int(cfg.get("forward_horizon_days", cfg.get("holding_horizon_days", 22)))


def miner_target_col(miner: str) -> str:
    return f"return_miner_fwd_{miner}"


def pred_field() -> str:
    return "pred_return_fwd"


def horizon_label(days: int | None = None) -> str:
    d = days or forward_horizon_days()
    if d <= 1:
        return "next JSE session (t+1)"
    return f"{d} trading sessions (~1 month)" if d >= 15 else f"{d} trading sessions"
