# Development

This guide is for contributors working on the `efficient-kan` source tree.

The package is intentionally small: the public API is `KANLinear`, `KAN`, and
`__version__`; the runtime dependency should remain PyTorch only unless there is
a compelling and explicitly reviewed reason to expand it.

## Setup

From the repository root:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e ".[dev]"
```

Use Python 3.10 or newer.

The project uses a pip and `pyproject.toml` workflow. It does not use PDM or a
checked-in lockfile.

## Repository Layout

```text
src/efficient_kan/
  __init__.py
  kan.py
  benchmark.py
  provenance.py
tests/
examples/
scripts/
docs/
```

Core implementation work belongs in `src/efficient_kan/kan.py`. Source-checkout
wrappers live in `scripts/`; reusable CLI logic lives in package modules.

## Local Validation

Run the full local gate before release candidates and before downstream
projects generate fresh research artifacts:

```bash
scripts/validate.sh
```

For targeted development, use smaller checks first:

```bash
python -m compileall src tests examples scripts
ruff check .
pytest -q
pytest --cov=src/efficient_kan --cov-report=term-missing tests
python scripts/benchmark.py --quick
python scripts/provenance.py --json
```

The repository intentionally does not use CI/CD. Do not add hosted workflow
files as part of ordinary package changes.

## Tests

Tests use `pytest`.

```bash
pytest -q
```

Coverage is configured in `pyproject.toml`:

```bash
pytest --cov=src/efficient_kan --cov-report=term-missing tests
```

Test guidance:

- prefer deterministic tests with fixed seeds;
- avoid long convergence tests in the default suite;
- test tensor shapes and finite values;
- test gradients for differentiable paths;
- test state-dict round trips when state behavior changes;
- test invalid inputs for public APIs;
- pin numerical contracts directly when possible.

Good numerical tests include:

- B-spline non-negativity and partition of unity on interior points;
- `curve2coeff()` reconstruction on well-conditioned inputs;
- strictly increasing grids after valid updates;
- no mutation after failed grid updates;
- dtype/device smoke checks;
- finite gradients.

## Linting

Ruff is the local linter:

```bash
ruff check .
```

Line length and target Python version are configured in `pyproject.toml`.

## Package Design Constraints

Keep the package focused:

- preserve the small public API;
- keep runtime dependencies limited to PyTorch;
- keep examples optional;
- keep benchmark and provenance tooling lightweight;
- avoid embedding downstream simulation orchestration in this package;
- avoid broad refactors when a narrow fix is sufficient.

`KANLinear` and `KAN` should continue to behave like ordinary PyTorch modules.
Buffers and parameters should move through normal `.to(device)` and dtype
conversion behavior.

## Numerical Contract Changes

Treat the following as public numerical behavior:

- spline basis construction;
- coefficient solves via `torch.linalg.lstsq`;
- initialization scales;
- shape semantics;
- adaptive grid updates;
- regularization semantics;
- dtype and device behavior;
- mutation behavior of `update_grid()`.

When proposing a behavior-changing numerical fix:

1. Describe the current behavior.
2. Explain why it is wrong or unsafe.
3. State the intended new behavior.
4. Add deterministic tests that fail under the old behavior when feasible.
5. Update relevant docs.
6. Add a changelog entry.
7. Run the local validation gate.

Separate semantic-preserving refactors from intentional numerical fixes. This
makes downstream research changes easier to audit.

## Backward Compatibility

Keep public APIs backward compatible unless a change fixes a documented bug or
rejects an invalid state. Prefer explicit validation errors over silent
behavior changes.

If an incompatibility is necessary, document it in:

- `CHANGELOG.md`;
- relevant files under `docs/`;
- tests that capture the new contract.

## Dependencies

Runtime dependencies live in `project.dependencies` in `pyproject.toml`. The
runtime dependency policy is:

```toml
dependencies = [
    "torch>=2.3.0",
]
```

Development, build, test, lint, and example tools belong in optional dependency
groups such as `dev` and `examples`.

Do not add a runtime dependency for convenience if the same result can be
implemented clearly with PyTorch or the Python standard library.

## Documentation

Update documentation when user-facing behavior changes.

Use:

- `README.md` for the concise overview;
- `docs/quickstart.md` for runnable examples;
- `docs/api.md` for public API details;
- topical docs for numerical contracts, grid updates, regularization,
  benchmarking, reproducibility, and release process;
- `CHANGELOG.md` for release-facing changes.

Keep documentation honest about numerical behavior. Do not describe benchmark
results, convergence, or estimator performance as general facts unless they are
backed by a specific artifact and context.

## Provenance and Benchmarks

Use provenance before downstream evidence runs:

```bash
python scripts/provenance.py --json --source-checkout
```

Use quick benchmarks for smoke checks:

```bash
python scripts/benchmark.py --quick
```

Use non-quick benchmarks for same-machine performance comparisons:

```bash
python scripts/benchmark.py
```

Benchmark results are environment-specific. Record the exact command and
provenance with any performance claim.
