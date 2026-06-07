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

optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
criterion = torch.nn.MSELoss()

for _ in range(100):
    optimizer.zero_grad()
    prediction = model(x)
    loss = criterion(prediction, y) + 1e-4 * model.regularization_loss(
        regularize_activation=1.0,
        regularize_entropy=0.0,
    )
    loss.backward()
    optimizer.step()

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
- `grid_size`: number of spline grid intervals.
- `spline_order`: B-spline order, with `3` corresponding to cubic splines.
- `base_activation`: activation applied before the base linear transformation.
- `enable_standalone_scale_spline`: enables a learnable scale on spline terms.
- `grid_eps`: interpolation between adaptive and uniform grids during grid
  updates.
- `grid_range`: initial spline-grid range.

Inputs should generally be scaled into a range compatible with `grid_range`,
especially when `update_grid=False`.

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
builds the package, checks package metadata, performs editable-install dry runs,
and checks whitespace. The repository intentionally uses a pip/`pyproject.toml`
workflow rather than PDM or a checked-in lockfile.

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
