# Changelog

All notable changes to this fork are documented here.

The format follows the spirit of Keep a Changelog, and this project uses
semantic versioning once releases are cut.

## [Unreleased]

### Added

- Added repo-local development instructions in `AGENTS.md`.
- Added deterministic tests for forward shapes, spline bases, coefficient
  interpolation, grid updates, regularization, initialization scaling, state
  dict round trips, and a tiny training smoke.
- Added pytest and ruff configuration in `pyproject.toml`.
- Added optional dependency groups for development and examples.
- Added GitHub Actions CI for compile, lint, tests, coverage, and package
  metadata checks.
- Added contributor documentation for pip/pyproject development and release
  checks.

### Changed

- Runtime dependencies now only include PyTorch; test and example dependencies
  moved to optional extras.
- Build configuration now uses setuptools via `pyproject.toml`; PDM lockfile
  support was removed.
- Supported Python metadata now targets Python 3.10 and newer, matching the CI
  matrix and modern setuptools packaging metadata.
- MNIST example now uses a `main()` function and import-safe module guard.
- `KAN` now forwards the `enable_standalone_scale_spline` option to its
  `KANLinear` layers.
- Initialization scale parameters now scale initialized tensors directly instead
  of being passed as the Kaiming negative-slope argument.

### Fixed

- `regularization_loss()` now remains finite when spline weights are all zero.
- User-facing shape/value validation now raises explicit exceptions instead of
  relying on Python assertions.

## [0.1.0]

### Added

- Initial forked package exposing `KANLinear` and `KAN`.
