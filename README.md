# efficient-kan

[![Python](https://img.shields.io/badge/python-%3E%3D3.10-blue)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-%3E%3D2.3-ee4c2c)](https://pytorch.org/)
[![Package](https://img.shields.io/badge/package-pyproject.toml-informational)](./pyproject.toml)

`efficient-kan` is a compact pure-PyTorch implementation of Kolmogorov-Arnold
Network (KAN) layers. It is designed for research workflows that need a small,
importable KAN dependency with predictable tensor behaviour, deterministic tests,
and minimal runtime dependencies.

The implementation focuses on the efficient formulation used in modern KAN
implementations: inputs are expanded over B-spline bases and then combined with
ordinary matrix operations. This avoids the large activation tensors used by
less efficient formulations and keeps forward and backward passes close to
standard PyTorch linear algebra.

## Why Use This Package?

- Pure PyTorch runtime dependency.
- Small public API: `KANLinear` and `KAN`.
- Efficient B-spline basis computation followed by matrix multiplication.
- Optional learnable spline scaling.
- Built-in spline-weight regularisation via `regularization_loss()`.
- Adaptive spline grid updates through `update_grid()`.
- Local validation with tests, linting, coverage, build checks, and metadata
  checks.
- Local provenance reporting for downstream research manifests.

This repository is also the KAN dependency used in:

> Shaw, C. (2025). "Model Risk in Machine-Learning Distributional IV Estimation".
> [arXiv:2506.12765](https://arxiv.org/abs/2506.12765)

## Installation

Install directly from GitHub:

```bash
python -m pip install "git+https://github.com/shawcharles/efficient-kan.git"
```

For local development:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e ".[dev]"
```

For the MNIST example:

```bash
python -m pip install -e ".[examples]"
python examples/mnist.py
```

## Quick Start

```python
import torch
from efficient_kan import KAN

model = KAN(
    layers_hidden=[10, 32, 1],
    grid_size=5,
    spline_order=3,
)

x = torch.randn(128, 10)
y = torch.randn(128, 1)

optimiser = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
criterion = torch.nn.MSELoss()

for _ in range(100):
    optimiser.zero_grad()
    prediction = model(x)
    loss = criterion(prediction, y) + 1e-4 * model.regularization_loss(
        regularize_activation=1.0,
        regularize_entropy=0.0,
    )
    loss.backward()
    optimiser.step()

with torch.no_grad():
    prediction = model(torch.randn(8, 10))
```

For binary classification, use a one-output model with
`torch.nn.BCEWithLogitsLoss`. For multiclass classification, use one output per
class with `torch.nn.CrossEntropyLoss`.

## API Overview

The package exposes two classes:

| Object | Purpose |
| --- | --- |
| `KANLinear` | A single KAN layer with base linear weights, spline weights, B-spline bases, optional spline scaling, and adaptive grid updates. |
| `KAN` | A stack of `KANLinear` layers constructed from a list of hidden dimensions. |

The main constructor is:

```python
KAN(
    layers_hidden,
    grid_size=5,
    spline_order=3,
    scale_noise=0.1,
    scale_base=1.0,
    scale_spline=1.0,
    enable_standalone_scale_spline=True,
    base_activation=torch.nn.SiLU,
    grid_eps=0.02,
    grid_range=(-1.0, 1.0),
)
```

Important parameters:

- `layers_hidden`: layer widths, including input and output dimensions, for
  example `[input_dim, hidden_dim, output_dim]`.
- `grid_size`: positive number of spline grid intervals.
- `spline_order`: positive B-spline order, with `3` corresponding to cubic
  splines.
- `scale_noise`, `scale_base`, `scale_spline`: finite scalar initialisation
  scales.
- `base_activation`: activation applied before the base linear transformation.
- `enable_standalone_scale_spline`: enables a learnable scale on spline terms.
- `grid_eps`: interpolation between adaptive and uniform grids during grid
  updates. It must be finite and in `[0, 1]`.
- `grid_range`: two increasing finite values defining the initial spline-grid
  range.

Inputs should generally be scaled into a range compatible with `grid_range`,
especially when `update_grid=False`.

### Shape Semantics

`KANLinear.forward()` and `KAN.forward()` accept tensors whose last dimension is
the input feature dimension. Any leading dimensions are treated as batch-like
dimensions and are restored on output:

```python
model = KAN([3, 4, 2])
x = torch.randn(5, 6, 3)
assert model(x).shape == (5, 6, 2)
```

Mutating grid updates follow the same convention:

```python
out = model(x, update_grid=True)
```

The lower-level `KANLinear.b_splines()` and `KANLinear.curve2coeff()` methods
operate on explicit two-dimensional tensors with shape
`(batch_size, in_features)`.

### Numerical Contracts

- Spline grids are strictly increasing after valid adaptive updates.
- `b_splines()` returns finite basis values. For in-range points on a valid
  grid, the basis is non-negative and sums to one up to floating-point
  tolerance.
- `curve2coeff()` uses `torch.linalg.lstsq`; it interpolates when the supplied
  spline basis has enough rank, otherwise it returns PyTorch's least-squares or
  minimum-norm solution.
- `update_grid()` mutates `grid` and `spline_weight`, then refits spline
  coefficients to preserve the spline contribution at the supplied samples up
  to numerical solve tolerance. This preservation accounts for
  `enable_standalone_scale_spline=True`.
- Inputs to numerical solve and grid-update paths must be finite. `margin` in
  `update_grid()` must be a finite positive scalar.
- Pure adaptive updates with `grid_eps=0` require enough variation in each
  feature to produce a strictly increasing grid. Degenerate batches fail fast
  before mutating model state.

KAN splines extrapolate outside the initial `grid_range` using the extended knot
grid. For stable research workflows, scale features into a range compatible with
the active grid and treat adaptive updates as explicit state mutation. A common
training pattern is to update grids periodically during early training and use
ordinary forward passes for evaluation.

## Regularisation

`regularization_loss()` returns an L1-type penalty on spline weights, with an
optional entropy term over the normalised spline-weight magnitudes.

```python
loss = task_loss + 1e-4 * model.regularization_loss(
    regularize_activation=1.0,
    regularize_entropy=0.0,
)
```

This formulation is chosen to preserve the efficient matrix-multiplication
implementation. It is not numerically identical to activation-based
regularisation from the original KAN paper, which depends on expanded
intermediate activations.

## Grid Updates

Each `KANLinear` layer supports adaptive grid updates:

```python
model(x, update_grid=True)
```

When `update_grid=True`, each layer updates its spline grid from the current
batch before computing the layer output. This can be useful during training, but
it mutates model state and is more expensive than a normal forward pass. Use it
deliberately, for example periodically rather than at every inference call.

## Development

Run the local validation gate from the repository root:

```bash
scripts/validate.sh
```

The gate compiles source files, runs Ruff, runs the test suite with coverage,
compiles scripts, runs the benchmark smoke check, emits a provenance smoke
report, builds the package, checks package metadata, performs editable-install
dry runs, and checks whitespace. This local gate is the authoritative readiness
check for the repository; the project intentionally does not use CI/CD or
hosted workflow files. The repository uses a pip/`pyproject.toml` workflow
rather than PDM or a checked-in lockfile.

For a lighter local performance smoke check:

```bash
python scripts/benchmark.py --quick
```

Installed distributions also expose the same benchmark as a console command:

```bash
efficient-kan-benchmark --quick
```

The benchmark uses `torch.utils.benchmark` and reports local timings for
`KANLinear.forward()`, full `KAN.forward()`, a backward pass, `update_grid()`,
and `regularization_loss()`. Results depend on the PyTorch version, BLAS or CUDA
backend, device, dtype, and model dimensions, so downstream projects should
record their own benchmark context when performance matters.

For machine-readable local provenance:

```bash
python scripts/provenance.py --json
```

Installed distributions expose the same provenance reporter as:

```bash
efficient-kan-provenance --json
```

The JSON report includes the package version, git commit, dirty status, Python
version, platform, PyTorch version, CUDA availability/version, and package
import path. By default, the script reports the package resolved by the active
Python environment and falls back to a checkout `src/` tree only when the
package is otherwise unavailable. Use `--source-checkout` from the source
wrapper when you deliberately want to report this checkout's source tree:

```bash
python scripts/provenance.py --json --source-checkout
```

For the installed console command, pass `--checkout-root` when source-checkout
provenance should come from a checkout other than the current git repository:

```bash
efficient-kan-provenance --json --source-checkout --checkout-root /path/to/efficient-kan
```

The JSON separates `package` import metadata from `source_checkout` metadata so
installed distributions and local checkouts are not confused. Downstream
research manifests should record at least the `efficient-kan` commit SHA, dirty
status, package version, import mode/path, Python version, PyTorch version,
device/CUDA context, validation command, and validation date before generating
fresh simulation or empirical evidence.

See [CONTRIBUTING.md](./CONTRIBUTING.md) for the contributor workflow and
[CHANGELOG.md](./CHANGELOG.md) for notable changes.

## Citation

If this package supports academic work, please cite the paper that motivated
this maintained research implementation:

```text
Shaw, C. (2025). "Model Risk in Machine-Learning Distributional IV Estimation".
arXiv:2506.12765.
```

Please also cite the original KAN paper where appropriate:

```text
Liu, Z. et al. (2024). "KAN: Kolmogorov-Arnold Networks".
arXiv:2404.19756.
```

## Acknowledgements

This implementation is developed in conversation with the broader KAN software
ecosystem. It draws on design ideas from
[`Blealtan/efficient-kan`](https://github.com/Blealtan/efficient-kan), and the
KAN methodology originates with the original
[`pykan`](https://github.com/KindXiaoming/pykan) project and KAN paper by Liu
et al. (2024).
