---
title: Quickstart
description: Minimal installation and usage examples for efficient-kan.
permalink: /quickstart/
---

# Quickstart

This guide shows the smallest useful patterns for installing `efficient-kan`
and using `KAN` or `KANLinear` in ordinary PyTorch training loops.

The examples use only PyTorch and `efficient-kan`. They are intentionally small
and deterministic so they can be copied into notebooks, scripts, or downstream
research smoke tests.

## Install

Install the package from GitHub:

```bash
python -m pip install "git+https://github.com/shawcharles/efficient-kan.git"
```

For local development from a source checkout:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e ".[dev]"
```

Check that the package imports:

```python
import torch
from efficient_kan import KAN, KANLinear

print(torch.__version__)
print(KAN)
print(KANLinear)
```

## Regression

Use a one-output `KAN` with a regression loss such as `torch.nn.MSELoss`.
Inputs should generally be scaled into a range compatible with `grid_range`,
which defaults to `(-1.0, 1.0)`.

```python
import torch
from efficient_kan import KAN


torch.manual_seed(123)

n = 256
x = torch.rand(n, 3) * 2.0 - 1.0
y = (
    torch.sin(torch.pi * x[:, [0]])
    + 0.5 * x[:, [1]] ** 2
    - 0.25 * x[:, [2]]
)

model = KAN(
    layers_hidden=[3, 16, 1],
    grid_size=5,
    spline_order=3,
)

optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
criterion = torch.nn.MSELoss()

for step in range(200):
    optimizer.zero_grad()
    prediction = model(x)
    task_loss = criterion(prediction, y)
    penalty = model.regularization_loss(
        regularize_activation=1.0,
        regularize_entropy=0.0,
    )
    loss = task_loss + 1e-4 * penalty
    loss.backward()
    optimizer.step()

model.eval()
with torch.no_grad():
    prediction = model(x[:5])
    print(prediction)
```

## Binary Classification

Use one output and pass raw logits to `torch.nn.BCEWithLogitsLoss`. Do not add a
sigmoid before the loss.

```python
import torch
from efficient_kan import KAN


torch.manual_seed(123)

n = 256
x = torch.rand(n, 2) * 2.0 - 1.0
target = ((x[:, 0] ** 2 + x[:, 1] ** 2) < 0.5).float().unsqueeze(1)

model = KAN([2, 12, 1], grid_size=5, spline_order=3)
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
criterion = torch.nn.BCEWithLogitsLoss()

for step in range(200):
    optimizer.zero_grad()
    logits = model(x)
    loss = criterion(logits, target)
    loss.backward()
    optimizer.step()

model.eval()
with torch.no_grad():
    logits = model(x[:5])
    probabilities = torch.sigmoid(logits)
    predicted_class = probabilities >= 0.5
    print(probabilities)
    print(predicted_class)
```

## Multiclass Classification

Use one output per class and pass raw logits to `torch.nn.CrossEntropyLoss`.
Targets should be integer class labels with shape `(batch_size,)`.

```python
import torch
from efficient_kan import KAN


torch.manual_seed(123)

n = 300
x = torch.rand(n, 2) * 2.0 - 1.0
target = torch.empty(n, dtype=torch.long)
target[x[:, 0] < -0.25] = 0
target[(x[:, 0] >= -0.25) & (x[:, 1] < 0.25)] = 1
target[(x[:, 0] >= -0.25) & (x[:, 1] >= 0.25)] = 2

model = KAN([2, 16, 3], grid_size=5, spline_order=3)
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
criterion = torch.nn.CrossEntropyLoss()

for step in range(200):
    optimizer.zero_grad()
    logits = model(x)
    loss = criterion(logits, target)
    loss.backward()
    optimizer.step()

model.eval()
with torch.no_grad():
    logits = model(x[:5])
    predicted_class = logits.argmax(dim=1)
    print(logits)
    print(predicted_class)
```

## Using KANLinear Directly

`KANLinear` is useful when you want a single KAN layer inside a larger PyTorch
module. It follows ordinary PyTorch module semantics and accepts tensors whose
last dimension is the feature dimension.

```python
import torch
from efficient_kan import KANLinear


torch.manual_seed(123)

layer = KANLinear(
    in_features=4,
    out_features=2,
    grid_size=5,
    spline_order=3,
)

x = torch.rand(32, 4) * 2.0 - 1.0
output = layer(x)

assert output.shape == (32, 2)
print(output[:3])
```

Leading dimensions are treated as batch-like dimensions and are restored on
output:

```python
x_panel = torch.rand(8, 10, 4) * 2.0 - 1.0
output_panel = layer(x_panel)

assert output_panel.shape == (8, 10, 2)
```

The lower-level basis and coefficient methods use explicit two-dimensional
inputs:

```python
basis = layer.b_splines(x)
assert basis.shape == (32, 4, layer.grid_size + layer.spline_order)
```

## Adaptive Grid Updates

`update_grid()` adapts a layer's spline grid to a batch of inputs and mutates
model state. Use it deliberately during training, not during ordinary
evaluation.

```python
import torch
from efficient_kan import KAN


torch.manual_seed(123)

model = KAN([3, 12, 1], grid_size=5, spline_order=3)
x = torch.rand(128, 3) * 2.0 - 1.0

model.train()
with torch.no_grad():
    model(x, update_grid=True)

prediction = model(x)
print(prediction.shape)
```

For reproducible statistical workflows, record whether grid updates were used,
the package version, PyTorch version, random seeds, device, dtype, and training
schedule.
