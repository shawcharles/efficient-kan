---
title: Regularization
description: Guidance for efficient-kan regularization_loss.
permalink: /regularization/
---

# Regularization

`regularization_loss()` returns an efficient spline-weight penalty that can be
added to ordinary PyTorch task losses.

It is available on both public model classes:

```python
penalty = layer.regularization_loss()
penalty = model.regularization_loss()
```

For `KANLinear`, the returned value is a scalar tensor for that layer. For
`KAN`, the returned value is the sum of the layer penalties across the stacked
model.

## What It Penalizes

For each `KANLinear` layer, the implementation computes:

```python
l1_like = layer.spline_weight.abs().mean(dim=-1)
activation_term = l1_like.sum()
```

The optional entropy term is computed from the normalized `l1_like` values:

```python
p = l1_like / activation_term
entropy_term = -torch.sum(p[p > 0] * p[p > 0].log())
```

The final penalty is:

```python
regularize_activation * activation_term + regularize_entropy * entropy_term
```

If all spline weights are zero, the entropy term is treated as zero so the
result remains finite.

## Arguments

```python
regularization_loss(
    regularize_activation: float = 1.0,
    regularize_entropy: float = 1.0,
)
```

| Argument | Meaning |
| --- | --- |
| `regularize_activation` | Weight on the L1-style spline-weight magnitude term. |
| `regularize_entropy` | Weight on the entropy term over normalized spline-weight magnitudes. |

Both arguments are multipliers on the returned penalty components. They are not
optimizer weight decay values.

## Efficient Proxy vs Activation-Based KAN Regularization

Some KAN formulations regularize expanded per-sample spline activations. That
requires materializing or tracking activation values for each sample, feature,
output, and spline basis element.

`efficient-kan` does not do that. Its regularization is based on spline weights,
not expanded per-sample activations.

This choice preserves the package's efficient matrix-multiplication
implementation:

- lower memory pressure;
- no need to retain large activation tensors for regularization;
- simple scalar penalty that works in standard PyTorch training loops.

The tradeoff is semantic: this penalty is not numerically identical to
activation-based regularization from the original KAN paper or implementations
that explicitly store expanded activations.

For statistical workflows, treat the regularization choice as part of the model
specification. Record whether it was used and the multiplier applied to the
penalty in the training objective.

## Regression Example

```python
import torch
from efficient_kan import KAN


torch.manual_seed(123)

x = torch.rand(256, 3) * 2.0 - 1.0
y = torch.sin(torch.pi * x[:, [0]]) + 0.5 * x[:, [1]] ** 2

model = KAN([3, 16, 1], grid_size=5, spline_order=3)
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
criterion = torch.nn.MSELoss()

regularization_strength = 1e-4

for step in range(200):
    optimizer.zero_grad()
    prediction = model(x)
    task_loss = criterion(prediction, y)
    penalty = model.regularization_loss(
        regularize_activation=1.0,
        regularize_entropy=0.0,
    )
    loss = task_loss + regularization_strength * penalty
    loss.backward()
    optimizer.step()
```

This example uses the activation-style component only and disables the entropy
component.

## Binary Classification Example

Use raw logits with `torch.nn.BCEWithLogitsLoss`.

```python
import torch
from efficient_kan import KAN


torch.manual_seed(123)

x = torch.rand(256, 2) * 2.0 - 1.0
y = ((x[:, 0] ** 2 + x[:, 1] ** 2) < 0.5).float().unsqueeze(1)

model = KAN([2, 12, 1], grid_size=5, spline_order=3)
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
criterion = torch.nn.BCEWithLogitsLoss()

regularization_strength = 5e-5

for step in range(200):
    optimizer.zero_grad()
    logits = model(x)
    task_loss = criterion(logits, y)
    penalty = model.regularization_loss(
        regularize_activation=1.0,
        regularize_entropy=0.0,
    )
    loss = task_loss + regularization_strength * penalty
    loss.backward()
    optimizer.step()
```

## Multiclass Classification Example

Use one output per class with `torch.nn.CrossEntropyLoss`.

```python
import torch
from efficient_kan import KAN


torch.manual_seed(123)

x = torch.rand(300, 2) * 2.0 - 1.0
y = torch.empty(300, dtype=torch.long)
y[x[:, 0] < -0.25] = 0
y[(x[:, 0] >= -0.25) & (x[:, 1] < 0.25)] = 1
y[(x[:, 0] >= -0.25) & (x[:, 1] >= 0.25)] = 2

model = KAN([2, 16, 3], grid_size=5, spline_order=3)
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
criterion = torch.nn.CrossEntropyLoss()

regularization_strength = 1e-4

for step in range(200):
    optimizer.zero_grad()
    logits = model(x)
    task_loss = criterion(logits, y)
    penalty = model.regularization_loss(
        regularize_activation=1.0,
        regularize_entropy=0.0,
    )
    loss = task_loss + regularization_strength * penalty
    loss.backward()
    optimizer.step()
```

## Entropy Term

The entropy term is available for users who want to penalize the distribution
of spline-weight magnitudes across output/input feature pairs.

```python
penalty = model.regularization_loss(
    regularize_activation=1.0,
    regularize_entropy=0.1,
)
```

Because the entropy term changes the objective, tune it deliberately. For
simple baselines and reproducible examples, start with:

```python
penalty = model.regularization_loss(
    regularize_activation=1.0,
    regularize_entropy=0.0,
)
```

and apply a separate outer multiplier such as `1e-4` when adding it to the task
loss.

## Interaction with Optimizer Weight Decay

`regularization_loss()` and optimizer weight decay are different mechanisms.

```python
optimizer = torch.optim.AdamW(
    model.parameters(),
    lr=1e-3,
    weight_decay=1e-4,
)
```

AdamW weight decay acts through the optimizer update. `regularization_loss()`
adds a differentiable scalar term to the training objective. You can use either
or both, but the combination changes the effective regularization policy and
should be documented in research artifacts.

## Practical Guidance

- Start with `regularize_entropy=0.0` unless the entropy term is part of the
  experiment.
- Use a small outer multiplier, for example `1e-5` to `1e-4`, and tune it as a
  model hyperparameter.
- Record the outer multiplier and both `regularization_loss()` arguments.
- Do not compare regularized and unregularized models without reporting the
  objective difference.
- Remember that this is spline-weight regularization, not expanded
  activation-based KAN regularization.
