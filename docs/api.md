---
title: API Reference
description: Public API reference for KAN and KANLinear.
permalink: /api/
---

# API Reference

`efficient-kan` exposes a deliberately small public API:

```python
from efficient_kan import KAN, KANLinear, __version__
```

`KANLinear` is the core layer. `KAN` is a stack of `KANLinear` layers.

The package follows ordinary PyTorch module semantics: modules are
`torch.nn.Module` instances, parameters are trainable tensors, buffers move with
`.to(device)` and dtype conversion, and gradients are handled by PyTorch
autograd except for explicit no-grad mutating grid updates.

## Version

```python
from efficient_kan import __version__
```

`__version__` is a string matching the package release version.

## KANLinear

```python
KANLinear(
    in_features: int,
    out_features: int,
    grid_size: int = 5,
    spline_order: int = 3,
    scale_noise: float = 0.1,
    scale_base: float = 1.0,
    scale_spline: float = 1.0,
    enable_standalone_scale_spline: bool = True,
    base_activation: type[torch.nn.Module] = torch.nn.SiLU,
    grid_eps: float = 0.02,
    grid_range: tuple[float, float] = (-1.0, 1.0),
)
```

`KANLinear` implements one KAN layer with two additive paths:

- a base linear path applied to `base_activation(x)`;
- a spline path formed from B-spline basis values and spline coefficients.

### Constructor Arguments

| Argument | Meaning | Constraints |
| --- | --- | --- |
| `in_features` | Number of input features. | Positive integer. |
| `out_features` | Number of output features. | Positive integer. |
| `grid_size` | Number of spline grid intervals. | Positive integer. |
| `spline_order` | B-spline order. `3` corresponds to cubic splines. | Positive integer. |
| `scale_noise` | Scale for random spline initialisation noise. | Finite scalar. |
| `scale_base` | Multiplicative scale applied to initialised base weights. | Finite scalar. |
| `scale_spline` | Multiplicative scale applied to initialised spline terms. | Finite scalar. |
| `enable_standalone_scale_spline` | Whether to use a separate learnable scale for spline coefficients. | Boolean-like value. |
| `base_activation` | PyTorch module class used before the base linear transformation. | Callable module class with a zero-argument constructor. |
| `grid_eps` | Interpolation between adaptive and uniform grid updates. `0` is purely adaptive; `1` is purely uniform. | Finite scalar in `[0, 1]`. |
| `grid_range` | Initial interior grid range. | Two increasing finite values. |

### State

After construction, the layer contains:

| Attribute | Type | Shape |
| --- | --- | --- |
| `base_weight` | `torch.nn.Parameter` | `(out_features, in_features)` |
| `spline_weight` | `torch.nn.Parameter` | `(out_features, in_features, grid_size + spline_order)` |
| `spline_scaler` | `torch.nn.Parameter` when enabled | `(out_features, in_features)` |
| `grid` | registered buffer | `(in_features, grid_size + 2 * spline_order + 1)` |

`grid` is a buffer, not a trainable parameter. It moves with the module when
using `.to(device)` or dtype conversion.

### forward

```python
output = layer(x)
```

Evaluates the layer.

| Input | Shape |
| --- | --- |
| `x` | `(..., in_features)` |

| Return | Shape |
| --- | --- |
| `output` | `(..., out_features)` |

Any leading dimensions are treated as batch-like dimensions and restored on
output:

```python
layer = KANLinear(4, 2)
x = torch.randn(8, 10, 4)
assert layer(x).shape == (8, 10, 2)
```

Expected errors:

- raises `ValueError` if `x` is scalar;
- raises `ValueError` if the last dimension of `x` is not `in_features`;
- may raise `ValueError` from the basis path if the flattened input contains no
  samples or non-finite values.

### b_splines

```python
basis = layer.b_splines(x)
```

Computes B-spline basis values on the current grid.

| Input | Shape |
| --- | --- |
| `x` | `(batch_size, in_features)` |

| Return | Shape |
| --- | --- |
| `basis` | `(batch_size, in_features, grid_size + spline_order)` |

For in-range points on a strictly increasing grid, basis values are
non-negative and sum to one up to floating-point tolerance.

Expected errors:

- raises `ValueError` if `x` is not two-dimensional with second dimension
  `in_features`;
- raises `ValueError` if `x` has zero rows;
- raises `ValueError` if `x` contains `NaN` or infinite values;
- raises `RuntimeError` if an internal basis shape invariant is violated.

### curve2coeff

```python
coefficients = layer.curve2coeff(x, y)
```

Solves B-spline coefficients for sample values using `torch.linalg.lstsq`.

| Input | Shape |
| --- | --- |
| `x` | `(batch_size, in_features)` |
| `y` | `(batch_size, in_features, out_features)` |

| Return | Shape |
| --- | --- |
| `coefficients` | `(out_features, in_features, grid_size + spline_order)` |

When the supplied basis has enough rank, the result interpolates the supplied
sample values up to numerical solve tolerance. Otherwise, the result follows
PyTorch's least-squares or minimum-norm behaviour for `torch.linalg.lstsq`.

Expected errors:

- raises `ValueError` if `x` is not two-dimensional with second dimension
  `in_features`;
- raises `ValueError` if `x` has zero rows;
- raises `ValueError` if `x` contains non-finite values;
- raises `ValueError` if `y` does not have shape
  `(batch_size, in_features, out_features)`;
- raises `ValueError` if `y` contains non-finite values;
- raises `RuntimeError` if `torch.linalg.lstsq` fails;
- raises `RuntimeError` if an internal coefficient shape invariant is violated.

### scaled_spline_weight

```python
scaled = layer.scaled_spline_weight
```

Property returning spline coefficients after applying the optional standalone
spline scaler.

| Return | Shape |
| --- | --- |
| `scaled` | `(out_features, in_features, grid_size + spline_order)` |

When `enable_standalone_scale_spline=True`, this is:

```python
layer.spline_weight * layer.spline_scaler.unsqueeze(-1)
```

When standalone scaling is disabled, this is `layer.spline_weight`.

### update_grid

```python
layer.update_grid(x, margin=0.01)
```

Adapts the spline grid to a batch of inputs and refits spline coefficients so
the spline contribution at the supplied samples is preserved up to least-squares
solve tolerance.

This method is decorated with `torch.no_grad()` and mutates layer state.

| Input | Shape |
| --- | --- |
| `x` | `(..., in_features)` |
| `margin` | finite positive scalar |

| Return |
| --- |
| `None` |

Mutated state:

- `grid`;
- `spline_weight`.

`spline_scaler` is not changed. If standalone spline scaling is enabled and a
scaler entry is zero, the update keeps the corresponding unscaled
`spline_weight` finite and leaves the scaled spline contribution at zero.

Expected errors:

- raises `ValueError` if `margin` is non-finite or not positive;
- raises `ValueError` if `x` is scalar;
- raises `ValueError` if the last dimension of `x` is not `in_features`;
- raises `ValueError` if the flattened input has zero rows;
- raises `ValueError` if `x` contains non-finite values;
- raises `ValueError` if existing spline outputs are non-finite before the
  update;
- raises `RuntimeError` if the proposed updated grid is not strictly
  increasing;
- raises `RuntimeError` if the coefficient refit fails.

If grid construction fails before the final state assignment, the existing
`grid` and `spline_weight` are left unchanged.

### regularization_loss

```python
penalty = layer.regularization_loss(
    regularize_activation=1.0,
    regularize_entropy=1.0,
)
```

Returns the efficient spline-weight regularisation penalty as a scalar tensor.

The activation component is based on mean absolute spline-weight magnitudes.
The optional entropy component is computed from the normalised magnitudes. This
preserves the memory-efficient implementation and is not identical to
regularizing expanded per-sample spline activations.

| Argument | Meaning |
| --- | --- |
| `regularize_activation` | Weight on the L1-style activation term. |
| `regularize_entropy` | Weight on the entropy term. |

| Return | Shape |
| --- | --- |
| `penalty` | scalar tensor |

## KAN

```python
KAN(
    layers_hidden: list[int],
    grid_size: int = 5,
    spline_order: int = 3,
    scale_noise: float = 0.1,
    scale_base: float = 1.0,
    scale_spline: float = 1.0,
    enable_standalone_scale_spline: bool = True,
    base_activation: type[torch.nn.Module] = torch.nn.SiLU,
    grid_eps: float = 0.02,
    grid_range: tuple[float, float] = (-1.0, 1.0),
)
```

`KAN` stacks `KANLinear` layers. The `layers_hidden` list contains the input
width, any hidden widths, and the output width.

Examples:

```python
KAN([3, 16, 1])      # 3 inputs, one hidden layer, 1 output
KAN([3, 32, 16, 2])  # 3 inputs, two hidden layers, 2 outputs
```

### Constructor Arguments

All constructor arguments except `layers_hidden` are passed through to each
`KANLinear` layer.

| Argument | Meaning | Constraints |
| --- | --- | --- |
| `layers_hidden` | Layer widths, including input and output widths. | At least two entries. Each adjacent pair must define a valid `KANLinear`. |
| `grid_size` | Number of spline grid intervals per layer. | Positive integer. |
| `spline_order` | B-spline order per layer. | Positive integer. |
| `scale_noise` | Scale for random spline initialisation noise. | Finite scalar. |
| `scale_base` | Multiplicative scale applied to initialised base weights. | Finite scalar. |
| `scale_spline` | Multiplicative scale applied to initialised spline terms. | Finite scalar. |
| `enable_standalone_scale_spline` | Whether each layer uses a separate learnable spline scale. | Boolean-like value. |
| `base_activation` | PyTorch module class used before each base linear transformation. | Callable module class with a zero-argument constructor. |
| `grid_eps` | Interpolation between adaptive and uniform grid updates. | Finite scalar in `[0, 1]`. |
| `grid_range` | Initial interior grid range for each layer. | Two increasing finite values. |

Expected constructor errors:

- raises `ValueError` if `layers_hidden` has fewer than two entries;
- raises the relevant `KANLinear` constructor error if any adjacent layer pair
  or shared spline/grid argument is invalid.

### State

| Attribute | Type | Meaning |
| --- | --- | --- |
| `layers` | `torch.nn.ModuleList` | Sequence of `KANLinear` layers. |
| `grid_size` | integer | Stored grid size. |
| `spline_order` | integer | Stored spline order. |

Each layer owns its own parameters and grid buffer.

### forward

```python
output = model(x, update_grid=False)
```

Evaluates the stacked network.

| Input | Shape |
| --- | --- |
| `x` | `(..., layers_hidden[0])` |
| `update_grid` | Boolean flag. |

| Return | Shape |
| --- | --- |
| `output` | `(..., layers_hidden[-1])` |

When `update_grid=False`, this is a standard differentiable forward pass.

When `update_grid=True`, each layer calls `layer.update_grid()` on that layer's
current input before evaluating the layer. This mutates every layer's grid and
spline weights in sequence. Use this during explicit adaptive-grid training
steps, not during ordinary evaluation.

Expected errors:

- raises `ValueError` if the input last dimension does not match the first
  layer's `in_features`;
- with `update_grid=True`, may raise any `KANLinear.update_grid()` error for the
  relevant layer;
- may raise a downstream `KANLinear.forward()` error if intermediate layer
  state becomes invalid.

### regularization_loss

```python
penalty = model.regularization_loss(
    regularize_activation=1.0,
    regularize_entropy=1.0,
)
```

Returns the sum of `regularization_loss()` across all layers.

| Return | Shape |
| --- | --- |
| `penalty` | scalar tensor |

## Shape Summary

| Object | Method | Input Shape | Return Shape |
| --- | --- | --- | --- |
| `KANLinear` | `forward(x)` | `(..., in_features)` | `(..., out_features)` |
| `KANLinear` | `b_splines(x)` | `(batch_size, in_features)` | `(batch_size, in_features, grid_size + spline_order)` |
| `KANLinear` | `curve2coeff(x, y)` | `(batch_size, in_features)`, `(batch_size, in_features, out_features)` | `(out_features, in_features, grid_size + spline_order)` |
| `KANLinear` | `update_grid(x)` | `(..., in_features)` | `None` |
| `KAN` | `forward(x)` | `(..., input_width)` | `(..., output_width)` |

## Public Contract Notes

- Inputs to `b_splines()`, `curve2coeff()`, and `update_grid()` must be finite.
- `forward()` accepts leading dimensions and restores them on output.
- `update_grid()` is an explicit state mutation.
- `grid` is required to be strictly increasing for valid spline behaviour.
- `curve2coeff()` delegates numerical solve behaviour to `torch.linalg.lstsq`.
- The importable package runtime dependency is PyTorch.
