---
title: Adaptive Grid Updates
description: Guidance for update_grid state mutation, reproducibility, and failure modes.
permalink: /grid-updates/
---

# Adaptive Grid Updates

`KANLinear.update_grid()` adapts a layer's spline grid to observed inputs.
`KAN.forward(update_grid=True)` applies the same operation layer by layer inside
a stacked `KAN` model.

Grid updates are useful, but they are not ordinary inference. They mutate model
state and should be treated as explicit training-time state transitions.

## What update_grid Does

For a `KANLinear` layer:

```python
layer.update_grid(x, margin=0.01)
```

the method:

1. flattens any leading dimensions of `x` into a sample dimension;
2. evaluates the current spline contribution at the supplied samples;
3. constructs a proposed grid from the empirical distribution of each feature;
4. blends the adaptive grid with a uniform grid according to `grid_eps`;
5. extends the grid for the spline order;
6. checks that the proposed grid is strictly increasing;
7. solves new spline coefficients on the proposed grid;
8. mutates `grid` and `spline_weight`.

The method returns `None`.

`update_grid()` is decorated with `torch.no_grad()`. It is not part of the
autograd graph, and it should not be interpreted as a differentiable operation.

## Mutated State

`update_grid()` mutates:

- `grid`, the registered knot-grid buffer;
- `spline_weight`, the trainable spline coefficient parameter.

It does not mutate:

- `base_weight`;
- `spline_scaler`;
- optimiser state.

When standalone spline scaling is enabled, the coefficient refit accounts for
`spline_scaler`. If a scaler entry is zero, the updated unscaled
`spline_weight` remains finite and the corresponding scaled spline contribution
stays zero.

## Shape Semantics

`update_grid()` follows the same leading-dimension convention as `forward()`:

```python
from efficient_kan import KANLinear
import torch


layer = KANLinear(4, 2)

x = torch.rand(32, 4)
layer.update_grid(x)

x_panel = torch.rand(8, 10, 4)
layer.update_grid(x_panel)
```

Both calls are valid. In the second call, the layer treats the input as
`80` samples with `4` features.

The lower-level spline methods `b_splines()` and `curve2coeff()` do not follow
this convention; they require explicit two-dimensional inputs.

## When to Use Grid Updates

Use grid updates when the spline grid should adapt to the empirical input
distribution seen during training. Common uses include:

- an initial adaptation pass before ordinary optimisation;
- periodic updates during early training;
- research experiments where adaptive grids are part of the estimator or model
  specification.

A conservative training pattern is:

```python
import torch
from efficient_kan import KAN


torch.manual_seed(123)

model = KAN([3, 16, 1], grid_size=5, spline_order=3)
optimiser = torch.optim.AdamW(model.parameters(), lr=1e-3)
criterion = torch.nn.MSELoss()

x = torch.rand(256, 3) * 2.0 - 1.0
y = torch.sin(torch.pi * x[:, [0]])

for step in range(200):
    model.train()

    if step in {0, 25, 50}:
        model(x, update_grid=True)

    optimiser.zero_grad()
    prediction = model(x)
    loss = criterion(prediction, y)
    loss.backward()
    optimiser.step()
```

This pattern makes grid adaptation explicit and keeps most training steps as
ordinary differentiable forward/backward passes.

## When Not to Use Grid Updates

Do not use `update_grid()` as a default inference path.

Avoid grid updates:

- during validation or test evaluation;
- inside prediction services;
- when comparing models unless each model follows the same documented update
  schedule;
- after loading a fitted model unless intentional post-load adaptation is part
  of the analysis;
- on batches that are too small or nearly degenerate in important features.

For evaluation, prefer:

```python
model.eval()
with torch.no_grad():
    prediction = model(x_eval)
```

not:

```python
model.eval()
with torch.no_grad():
    prediction = model(x_eval, update_grid=True)
```

The second form changes the fitted model before producing predictions.

## Reproducibility Implications

Because grid updates mutate state, they are part of the model specification and
training schedule. For statistical or econometric workflows, record:

- package version;
- PyTorch version;
- random seeds;
- device and dtype;
- `grid_size`, `spline_order`, `grid_eps`, `grid_range`, and `margin`;
- whether standalone spline scaling was enabled;
- exactly when grid updates occurred;
- whether updates used training data only or included validation/test data.

Using validation or test data to update grids changes the fitted model and can
leak evaluation information into the model state.

Saved model state includes the current `grid` buffer and spline coefficients.
Loading a state dict restores the adapted grids as long as the target model has
the same architecture.

## grid_eps and margin

`grid_eps` is set at construction time:

```python
layer = KANLinear(4, 2, grid_eps=0.02)
```

It controls the blend between two candidate grids:

- `grid_eps=0.0`: purely adaptive empirical grid;
- `grid_eps=1.0`: purely uniform grid over the observed feature range plus
  margin;
- values between `0.0` and `1.0`: convex blend of adaptive and uniform grids.

The default `0.02` keeps the grid close to adaptive while preserving a small
uniform component.

`margin` is passed to `update_grid()`:

```python
layer.update_grid(x, margin=0.01)
```

It expands the observed feature range before constructing the uniform grid
component. `margin` must be finite and positive.

## Failure Modes

`update_grid()` validates inputs before committing the new state.

Expected failures include:

- `ValueError` if `margin` is not finite or not positive;
- `ValueError` if `x` is scalar or its last dimension does not match
  `in_features`;
- `ValueError` if the flattened input contains zero samples;
- `ValueError` if `x` contains `NaN` or infinite values;
- `ValueError` if the existing spline outputs are non-finite before the update;
- `RuntimeError` if the proposed grid is not strictly increasing;
- `RuntimeError` if the least-squares coefficient refit fails.

Degenerate batches are the main practical failure mode. With a purely adaptive
grid, repeated or nearly repeated values can fail to produce a strictly
increasing grid:

```python
from efficient_kan import KANLinear
import torch


layer = KANLinear(2, 3, grid_eps=0.0)
x = torch.zeros(16, 2)

try:
    layer.update_grid(x)
except RuntimeError as exc:
    print(exc)
```

For this reason, adaptive updates should use batches with enough variation in
each feature. If degenerate batches are plausible, keep `grid_eps > 0`, increase
`margin`, update on larger batches, or skip updates for those batches.

## State Safety

The update path computes the proposed grid and new coefficients before assigning
them to layer state. If grid construction or coefficient solving fails before
assignment, the existing `grid` and `spline_weight` are left unchanged.

Once assignment succeeds, the mutation is intentional and persistent. It affects
subsequent forward passes, saved state dicts, and downstream predictions.

## KAN.forward(update_grid=True)

For a stacked model:

```python
output = model(x, update_grid=True)
```

each layer updates its grid using the current input to that layer, then
evaluates the layer. The next layer sees the transformed output of the previous
layer, not the original raw input.

This means `KAN.forward(update_grid=True)` mutates every layer in sequence. Use
it only when that full-model adaptation is intended.
