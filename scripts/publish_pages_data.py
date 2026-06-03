"""Copy latest signals into docs/ for GitHub Pages."""
from __future__ import annotations

import json
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SRC = ROOT / "data" / "forward_model" / "latest_signals.json"
DOCS = ROOT / "docs"
DST = DOCS / "signals.json"


def main() -> int:
    if not SRC.exists():
        print(f"Missing {SRC} — run generate_daily_signals.py first", file=sys.stderr)
        return 1
    DOCS.mkdir(parents=True, exist_ok=True)
    shutil.copy(SRC, DST)
    (DOCS / ".nojekyll").touch(exist_ok=True)
    meta = json.loads(DST.read_text(encoding="utf-8"))
    print(f"Published {DST} (signal_date={meta.get('signal_date')})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
