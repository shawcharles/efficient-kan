---
title: efficient-kan Documentation
---

# efficient-kan Documentation

`efficient-kan` is a compact pure-PyTorch implementation of
Kolmogorov-Arnold Network layers. It is designed for research workflows that
need a small KAN dependency with predictable tensor behavior, explicit numerical
contracts, local validation, and minimal runtime dependencies.

Use this site if you are:

- training `KAN` or `KANLinear` modules in PyTorch;
- using `efficient-kan` inside statistical or econometric research code;
- maintaining the package's numerical behavior, documentation, or release
  process.

## Start Here

- [Quickstart](quickstart.md): install commands and minimal examples for
  regression, binary classification, multiclass classification, and direct
  `KANLinear` use.
- [API Reference](api.md): public constructors, tensor shapes, return values,
  mutating methods, and expected errors.

## Numerical Behavior

- [Grid Updates](grid-updates.md): how `update_grid()` mutates state, when to
  use it, when to avoid it, and how to handle degenerate batches.
- [Regularization](regularization.md): `regularization_loss()`, its efficient
  spline-weight penalty, and examples for adding it to training objectives.
- [Reproducibility](reproducibility.md): seeds, PyTorch version sensitivity,
  provenance commands, and downstream artifact expectations.

## Local Tooling

- [Benchmarking](benchmarking.md): local benchmark commands, measured
  operations, limitations, and same-machine comparison guidance.
- [Development](development.md): editable install, tests, linting, coverage,
  package constraints, dependency policy, and numerical-change process.
- [Release Checklist](release-checklist.md): local-only release gate, version
  updates, changelog expectations, build checks, wheel smoke tests, provenance,
  and the no-CI/CD assumption.

## Repository

- [GitHub repository](https://github.com/shawcharles/efficient-kan)
- [Top-level README](https://github.com/shawcharles/efficient-kan#readme)

## Local Validation

From the repository root:

```bash
scripts/validate.sh
```

This local gate is the authoritative readiness check for the repository.
