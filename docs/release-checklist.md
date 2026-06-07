---
title: Release Checklist
description: Local-only release readiness gate for efficient-kan.
permalink: /release-checklist/
nav_order: 9
---

# Release Checklist

This repository uses a local-only release gate. It intentionally does not use
CI/CD or hosted workflow files.

Run release checks from the repository root.

## 1. Confirm Scope

Before cutting a release candidate:

- review the diff;
- confirm no local handoff or planning files are tracked;
- confirm `AGENTS.md` remains ignored and untracked;
- confirm runtime dependencies remain minimal;
- separate semantic-preserving refactors from intentional numerical fixes.

Useful commands:

```bash
git status --short
git diff --stat
git diff --check
```

## 2. Update Version Metadata

Update all version references for the release:

- `pyproject.toml`;
- `src/efficient_kan/__init__.py`;
- validation assertions in `scripts/validate.sh`;
- documentation examples if they mention the previous release;
- `CHANGELOG.md`.

The package version and `efficient_kan.__version__` should agree.

## 3. Update Changelog

Move relevant entries from `[Unreleased]` into a dated release section:

```markdown
## [0.2.0] - 2026-06-07
```

Call out:

- public API changes;
- numerical contract changes;
- bug fixes;
- validation or packaging changes;
- documentation additions;
- dependency changes.

If a change intentionally alters numerical behaviour, say so directly.

## 4. Run the Local Validation Gate

The authoritative release gate is:

```bash
scripts/validate.sh
```

The gate currently:

- removes stale build artefacts;
- compiles `src`, `tests`, `examples`, and `scripts`;
- runs Ruff;
- runs the test suite;
- runs coverage with the configured threshold;
- runs the benchmark smoke check;
- emits a provenance smoke report;
- builds the source distribution and wheel;
- runs `twine check`;
- inspects expected sdist and wheel contents;
- installs the built wheel into a temporary virtual environment;
- checks `__version__` from the installed wheel;
- runs installed `efficient-kan-provenance --json`;
- runs installed `efficient-kan-benchmark --quick`;
- dry-runs editable installs for base and dev extras;
- runs `git diff --check`.

Do not publish release artefacts unless this gate passes.

## 5. Manual Build and Metadata Checks

The validation gate runs these commands, but they are useful when diagnosing
release issues:

```bash
rm -rf build dist src/*.egg-info
python -m build
python -m twine check dist/*
```

If documentation changed, also render the GitHub Pages site locally:

```bash
cd docs
bundle install
bundle exec jekyll build
```

This is a local documentation check, not CI/CD.

Expected artefacts for version `0.2.0` are:

```text
dist/efficient_kan-0.2.0.tar.gz
dist/efficient_kan-0.2.0-py3-none-any.whl
```

Adjust filenames for future versions.

## 6. Installed-Wheel Smoke Test

The release gate installs the built wheel into a temporary virtual environment.
For a manual check:

```bash
tmpdir=$(mktemp -d)
python -m venv --system-site-packages "$tmpdir/venv"
"$tmpdir/venv/bin/python" -m pip install \
  dist/efficient_kan-0.2.0-py3-none-any.whl \
  --force-reinstall \
  --no-deps
"$tmpdir/venv/bin/python" - <<'PY'
from efficient_kan import KAN, __version__

assert __version__ == "0.2.0"
model = KAN([2, 3, 1])
print(__version__, model)
PY
"$tmpdir/venv/bin/efficient-kan-provenance" --json
"$tmpdir/venv/bin/efficient-kan-benchmark" --quick
rm -rf "$tmpdir"
```

Use the current release version in the wheel filename and assertion.

## 7. Provenance Check

Before tagging or handing off a release candidate, save local provenance:

```bash
python scripts/provenance.py --json --source-checkout
```

Confirm:

- package version is the intended release version;
- git commit is the intended release commit;
- dirty status is understood;
- import path is the intended source checkout or installed package;
- Python and PyTorch versions are recorded.

For final release artefacts, prefer a clean git status.

## 8. Package Contents

The source distribution should include:

- `README.md`;
- `CHANGELOG.md`;
- `CONTRIBUTING.md`;
- `LICENSE`;
- `pyproject.toml`;
- `docs/*.md`;
- `scripts/*.py` and `scripts/*.sh`;
- `tests/*.py`;
- `examples/*.py`;
- package source under `src/efficient_kan`.

The source distribution should not include:

- `handoff/`;
- `.planning/`;
- build artefacts;
- local caches;
- ignored agent-local files.

The wheel should include importable package modules and console entry points:

- `efficient_kan`;
- `efficient-kan-provenance`;
- `efficient-kan-benchmark`.

## 9. No CI/CD Assumption

Do not rely on GitHub Actions, hosted CI, or external automation to establish
release readiness. The local gate is the release gate.

If someone adds CI/CD in a downstream fork, it is supplemental and does not
replace this repository's local validation checklist.

## 10. Final Pre-Release Questions

Before tagging or publishing:

- Did all tests pass locally?
- Did coverage pass locally?
- Did build and `twine check` pass?
- Did the installed wheel import and run console commands?
- Is `CHANGELOG.md` current?
- Do version strings agree?
- Is provenance captured?
- Are numerical behaviour changes documented?
- Are runtime dependencies still limited to PyTorch?
