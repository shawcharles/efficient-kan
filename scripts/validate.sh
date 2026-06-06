#!/usr/bin/env bash
set -euo pipefail

python -m compileall src tests examples
ruff check .
pytest -q
pytest --cov=src/efficient_kan --cov-report=term-missing tests

rm -rf build dist src/*.egg-info
python -m build
python -m twine check dist/*

python -m pip install -e . --dry-run
python -m pip install -e '.[dev]' --dry-run
git diff --check
