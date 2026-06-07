#!/usr/bin/env bash
set -euo pipefail

rm -rf build dist src/*.egg-info

python -m compileall src tests examples scripts
ruff check .
pytest -q
pytest --cov=src/efficient_kan --cov-report=term-missing tests
python scripts/benchmark.py --quick
python scripts/provenance.py --json

rm -rf build dist src/*.egg-info
python -m build
python -m twine check dist/*
python - <<'PY'
import tarfile
import zipfile
from pathlib import Path

sdist = next(Path("dist").glob("efficient_kan-0.2.0.tar.gz"))
wheel = next(Path("dist").glob("efficient_kan-0.2.0-py3-none-any.whl"))

with tarfile.open(sdist) as archive:
    names = set(archive.getnames())
    assert any(name.endswith("README.md") for name in names)
    assert any(name.endswith("CHANGELOG.md") for name in names)
    assert any(name.endswith("CONTRIBUTING.md") for name in names)
    assert any(name.endswith("LICENSE") for name in names)
    assert any(name.endswith("tests/test_kan.py") for name in names)
    assert any(name.endswith("scripts/provenance.py") for name in names)
    assert any(name.endswith("scripts/benchmark.py") for name in names)
    assert any(name.endswith("docs/index.md") for name in names)
    assert any(name.endswith("docs/_config.yml") for name in names)
    assert not any("/handoff/" in name for name in names)

with zipfile.ZipFile(wheel) as archive:
    names = set(archive.namelist())
    assert "efficient_kan/__init__.py" in names
    assert "efficient_kan/provenance.py" in names
    assert "efficient_kan/benchmark.py" in names
    entry_points = [
        archive.read(name).decode()
        for name in names
        if name.endswith("entry_points.txt")
    ]
    assert entry_points
    assert "efficient-kan-provenance" in entry_points[0]
    assert "efficient-kan-benchmark" in entry_points[0]
PY

tmpdir=$(mktemp -d)
trap 'rm -rf "$tmpdir"' EXIT
python -m venv --system-site-packages "$tmpdir/venv"
"$tmpdir/venv/bin/python" -m pip install \
    dist/efficient_kan-0.2.0-py3-none-any.whl \
    --force-reinstall \
    --no-deps
"$tmpdir/venv/bin/python" - <<'PY'
from efficient_kan import __version__

assert __version__ == "0.2.0"
PY
"$tmpdir/venv/bin/efficient-kan-provenance" --json
"$tmpdir/venv/bin/efficient-kan-benchmark" --quick

python -m pip install -e . --dry-run
python -m pip install -e '.[dev]' --dry-run
git diff --check
