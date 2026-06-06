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

## Validation

Run the local validation gate:

```bash
scripts/validate.sh
```

The coverage gate is configured in `pyproject.toml`.

## Testing Guidelines

- Prefer deterministic unit tests with fixed seeds.
- Avoid long convergence tests in the default suite.
- Cover tensor shapes, finite outputs, gradients, serialization, and input
  validation for public APIs.
- Keep examples import-safe by guarding executable work with
  `if __name__ == "__main__":`.

## Release Checks

Before tagging or publishing a release candidate:

```bash
rm -rf build dist src/*.egg-info
python -m build
python -m twine check dist/*
python -m pip install dist/*.whl --force-reinstall
python -c "from efficient_kan import KAN; print(KAN([2, 3, 1]))"
```

Do not publish release artifacts until the local validation gate passes.
