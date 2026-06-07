# Contributing

This fork is used as the KAN dependency for downstream research code, so small
numerical changes can affect published simulation and empirical results.

## Development Setup

Use the pip/pyproject workflow from the repository root:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e ".[dev]"
```

Use Python 3.10 or newer.

The project does not use PDM or a lockfile. Runtime dependencies belong in
`project.dependencies`; test, lint, build, and example tooling belongs in
optional dependency groups.

This repository intentionally relies on local validation rather than CI/CD. Do
not add `.github` workflows or hosted automation as part of ordinary
contributions.

## Validation

Run the local validation gate:

```bash
scripts/validate.sh
```

The coverage gate is configured in `pyproject.toml`.

For targeted work on the core layer, run the bounded checks first:

```bash
python -m compileall src tests examples scripts
ruff check .
pytest -q
python scripts/benchmark.py --quick
python scripts/provenance.py --json
python -m efficient_kan.benchmark --quick
python -m efficient_kan.provenance --json
```

Use the full local gate before release candidates or before downstream projects
generate fresh research evidence from a new commit. Treat the output of
`scripts/validate.sh` as the authoritative readiness signal.

## Testing Guidelines

- Prefer deterministic unit tests with fixed seeds.
- Avoid long convergence tests in the default suite.
- Cover tensor shapes, finite outputs, gradients, serialization, and input
  validation for public APIs.
- Pin numerical contracts directly when possible: B-spline partition of unity,
  non-negativity, coefficient reconstruction, monotone grids after updates, and
  dtype/device smoke checks are preferred over broad convergence assertions.
- Keep examples import-safe by guarding executable work with
  `if __name__ == "__main__":`.

## Numerical Contract Changes

Treat spline bases, coefficient solves, adaptive grid updates, initialisation
scales, and regularisation semantics as public behaviour. If a change affects one
of those contracts:

- document the intended semantic change in `CHANGELOG.md`;
- add deterministic tests that would fail under the old buggy behaviour when
  feasible;
- update README API notes if user-facing input ranges, shape semantics, or
  mutation behaviour change;
- keep runtime dependencies limited to PyTorch unless the maintainer explicitly
  approves an expansion.

## Downstream Provenance

Downstream statistical projects should record local provenance before running
fresh simulations, empirical analyses, or benchmarks. At minimum, manifests
should include:

- `efficient-kan` commit SHA and dirty status;
- package version and import path;
- Python version and platform;
- PyTorch version plus device and CUDA availability/version;
- validation command and validation date;
- benchmark command when performance results are cited.

Use the JSON provenance command as the source for package, git, Python,
platform, PyTorch, and CUDA fields. By default it reports the package import
resolved by the active Python environment and falls back to this checkout's
`src/` tree only if needed:

```bash
python scripts/provenance.py --json
```

Use `python scripts/provenance.py --json --source-checkout` when the manifest
should explicitly describe this source checkout. The JSON distinguishes
`package.import_mode` and `package.import_path` from `source_checkout` metadata.

## Release Checks

Before tagging or publishing a release candidate:

```bash
rm -rf build dist src/*.egg-info
python -m build
python -m twine check dist/*
python -m pip install dist/*.whl --force-reinstall
python -c "from efficient_kan import KAN, __version__; print(__version__, KAN([2, 3, 1]))"
efficient-kan-provenance --json
efficient-kan-benchmark --quick
```

The source distribution should include contributor and release-facing files
(`README.md`, `CHANGELOG.md`, `CONTRIBUTING.md`, `LICENSE`), tests, examples,
and the local `scripts/` wrappers. The wheel should include the package modules
and console entry points for `efficient-kan-provenance` and
`efficient-kan-benchmark`.

For a non-installing artefact inspection:

```bash
python - <<'PY'
import tarfile, zipfile
from pathlib import Path
sdist = next(Path("dist").glob("efficient_kan-0.2.0.tar.gz"))
wheel = next(Path("dist").glob("efficient_kan-0.2.0-py3-none-any.whl"))
with tarfile.open(sdist) as archive:
    names = set(archive.getnames())
    assert any(name.endswith("README.md") for name in names)
    assert any(name.endswith("CHANGELOG.md") for name in names)
    assert any(name.endswith("CONTRIBUTING.md") for name in names)
    assert any(name.endswith("scripts/provenance.py") for name in names)
    assert any(name.endswith("scripts/benchmark.py") for name in names)
with zipfile.ZipFile(wheel) as archive:
    names = set(archive.namelist())
    assert "efficient_kan/__init__.py" in names
    assert any(name.endswith("entry_points.txt") for name in names)
PY
```

Do not publish release artefacts until the local validation gate passes.
