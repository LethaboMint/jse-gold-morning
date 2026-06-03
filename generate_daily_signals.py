"""
Daily signal generator entry point (for Task Scheduler / cron).

Reads signal_config.json, runs the model, writes latest snapshot + log.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
CONFIG_PATH = ROOT / "signal_config.json"


def main() -> int:
    if not CONFIG_PATH.exists():
        print(f"Missing config: {CONFIG_PATH}")
        return 1

    cfg = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
    sys.path.insert(0, str(ROOT))
    from score_miners_forward import run_generation

    try:
        run_generation(
            min_pred=float(cfg.get("min_pred", 0.0)),
            rules=str(cfg.get("rules", "production")),
            skip_duplicate=bool(cfg.get("skip_duplicate_same_day", True)),
            write_latest=bool(cfg.get("write_latest_snapshot", True)),
        )
    except Exception as e:
        print(f"ERROR: {e}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
