# AGENTS.md

This repository contains the active fork of `efficient-kan`, a compact PyTorch
implementation of Kolmogorov-Arnold Network layers used by downstream research
code such as `kan-d-iv-late`.

## Scope

Use this file as the repo-local guide for future agents working in:

- `/home/user/Documents/GITHUB/shawcharles/efficient-kan`

## Canonical Paths

- Package source: `src/efficient_kan/`
- Core implementation: `src/efficient_kan/kan.py`
- Public exports: `src/efficient_kan/__init__.py`
- Tests: `tests/`
- Examples: `examples/`
- Planning/reviews: `.planning/`

## Current Role

This repo is the lower-level KAN dependency for the D-IV-LATE paper project.
Treat numerical behavior as part of the public contract. Changes to spline
bases, coefficient interpolation, grid updates, initialization, or
regularization can change downstream empirical results.

## Working Rules

- Keep the importable runtime dependency surface small; the core library should
  only require PyTorch unless there is a strong reason to add more.
- Put test and example dependencies in optional dependency groups, not runtime
  dependencies.
- Prefer deterministic tests over convergence tests. If a training-convergence
  test is needed, mark it slow and use fixed seeds.
- Do not use progress bars or print large tensors in tests.
- Keep examples guarded by `if __name__ == "__main__":` so they can be imported
  safely.
- Replace assertion-based user input validation with explicit exceptions.
- Preserve backward-compatible constructor defaults unless a numerical bug fix
  requires changing them, and document any semantic change in `CHANGELOG.md`.

## Validation Commands

Run these from the repo root:

```bash
python -m compileall src tests examples
pytest -q
pytest --cov=src/efficient_kan --cov-report=term-missing tests
ruff check .
```

## Downstream Compatibility

Before using this library for fresh `kan-d-iv-late` evidence generation, confirm:

- `pip install -e .` works in the target environment.
- `pytest -q` passes.
- A small real `KAN` forward/backward/update-grid smoke test passes.
- The downstream repo records the `efficient-kan` commit SHA in its manifests.
