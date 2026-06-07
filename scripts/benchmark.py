#!/usr/bin/env python
"""Source-checkout wrapper for efficient-kan microbenchmarks.

The script intentionally uses only PyTorch and the package itself. Results are
environment-specific; use them to compare local changes on the same machine, not
as portable performance claims.
"""

from __future__ import annotations

import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if SRC.exists():
    sys.path.insert(0, str(SRC))

from efficient_kan.benchmark import main  # noqa: E402


if __name__ == "__main__":
    main()
