"""Publish all dashboard JSON into docs/ for GitHub Pages."""
from __future__ import annotations

import json
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DOCS = ROOT / "docs"
FM = ROOT / "data" / "forward_model"

SIGNALS_SRC = FM / "latest_signals.json"
SITE_DATA = DOCS / "site_data.json"


def load_json(path: Path) -> dict | list | None:
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def main() -> int:
    if not SIGNALS_SRC.exists():
        print(f"Missing {SIGNALS_SRC} — run generate_daily_signals.py first", file=sys.stderr)
        return 1

    DOCS.mkdir(parents=True, exist_ok=True)
    (DOCS / ".nojekyll").touch(exist_ok=True)

    signals = load_json(SIGNALS_SRC)
    audit = load_json(DOCS / "audit.json")
    performance = load_json(DOCS / "performance.json")

    shutil.copy(SIGNALS_SRC, DOCS / "signals.json")

    bundle = {
        "signals": signals,
        "audit": audit,
        "performance": performance,
    }
    SITE_DATA.write_text(json.dumps(bundle, indent=2), encoding="utf-8")

    print(f"Published {DOCS / 'signals.json'} (signal_date={signals.get('signal_date')})")
    print(f"Published {SITE_DATA} (audit rows={len((audit or {}).get('rows', []))}, perf days={len((performance or {}).get('daily', []))})")
    if not audit:
        print("WARN: audit.json missing — run: python audit_forward_log.py", file=sys.stderr)
    if not performance:
        print("WARN: performance.json missing — run: python audit_forward_log.py", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
