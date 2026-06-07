# Changelog

All notable changes to this fork are documented here.

The format follows the spirit of Keep a Changelog, and this project uses
semantic versioning once releases are cut.

## [Unreleased]

## [0.2.0] - 2026-06-07

### Added

- Added local agent guidance for maintainers; `AGENTS.md` remains ignored and
  untracked rather than part of release artefacts.
- Added `efficient_kan.__version__` and bumped package metadata to `0.2.0`.
- Added deterministic tests for forward shapes, spline bases, coefficient
  interpolation, grid updates, regularisation, initialisation scaling, state
  dict round trips, and a tiny training smoke.
- Added deterministic numerical invariant tests for B-spline non-negativity and
  partition of unity, `curve2coeff()` reconstruction, monotone adaptive grids,
  invalid numerical inputs, leading-dimension grid updates, and CPU
  double-precision forward/backward/update coverage.
- Added pytest and ruff configuration in `pyproject.toml`.
- Added optional dependency groups for development and examples.
- Added `scripts/validate.sh` for local compile, lint, tests, coverage,
  package build, and package metadata checks.
- Added `scripts/benchmark.py` for local `torch.utils.benchmark` smoke checks.
- Added `scripts/provenance.py` for JSON package, git, Python, platform,
  PyTorch, CUDA, and import-path provenance reports.
- Added installed console commands `efficient-kan-benchmark` and
  `efficient-kan-provenance`.
- Added explicit source-distribution manifest rules for release docs, tests,
  examples, and scripts while excluding local handoff state.
- Added contributor documentation for pip/pyproject development and release
  checks.
- Added project classifiers and project URLs to package metadata.

### Changed

- Moved reusable benchmark and provenance logic into package modules while
  keeping the top-level scripts as source-checkout wrappers.
- Runtime dependencies now only include PyTorch; test and example dependencies
  moved to optional extras.
- Build configuration now uses setuptools via `pyproject.toml`; PDM lockfile
  support was removed.
- Supported Python metadata now targets Python 3.10 and newer, matching modern
  setuptools packaging metadata.
- MNIST example now uses a `main()` function and import-safe module guard.
- `KAN` now forwards the `enable_standalone_scale_spline` option to its
  `KANLinear` layers.
- Initialisation scale parameters now scale initialised tensors directly instead
  of being passed as the Kaiming negative-slope argument.
- Public documentation now describes constructor ranges, shape semantics,
  adaptive-grid mutation, numerical invariants, and local benchmark usage.
- The local validation gate now compiles scripts and runs benchmark and
  provenance smoke checks before package build validation.

### Fixed

- `grid_eps` now rejects non-finite values and values outside `[0, 1]` before
  they can corrupt adaptive spline grids.
- `scale_noise`, `scale_base`, and `scale_spline` now reject non-finite values.
- `KAN.forward(update_grid=True)` now supports inputs with leading dimensions,
  matching the non-mutating forward path.
- `KANLinear.update_grid()` now accounts for standalone spline scaling when
  refitting coefficients, preserving the spline contribution at update samples
  for both scaling modes.
- Parameter updates now avoid `.data` mutation and use in-place operations under
  `torch.no_grad()`.
- `regularization_loss()` now remains finite when spline weights are all zero.
- User-facing shape/value validation now raises explicit exceptions instead of
  relying on Python assertions.

## [0.1.0]

### Added

- Initial forked package exposing `KANLinear` and `KAN`.
