#!/usr/bin/env python
"""Source-checkout wrapper for efficient-kan provenance reporting."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if SRC.exists():
    sys.path.insert(0, str(SRC))

from efficient_kan.provenance import main as provenance_main  # noqa: E402


def main() -> None:
    provenance_main([*sys.argv[1:], "--checkout-root", str(ROOT)])


if __name__ == "__main__":
    main()
