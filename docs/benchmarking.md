---
title: Benchmarking
description: Local benchmarking command and responsible comparison guidance.
permalink: /benchmarking/
---

# Benchmarking

`efficient-kan` includes a small local microbenchmark command for checking
performance-sensitive changes on the same machine.

The benchmark is a diagnostic tool. It is not a portable claim about package
speed across machines, PyTorch builds, BLAS libraries, CUDA versions, or model
sizes.

## Commands

From a source checkout:

```bash
python scripts/benchmark.py --quick
```

From an installed package:

```bash
efficient-kan-benchmark --quick
```

For longer timings, omit `--quick`:

```bash
python scripts/benchmark.py
```

## Options

```bash
python scripts/benchmark.py \
  --device cpu \
  --dtype float32 \
  --batch-size 128 \
  --in-features 16 \
  --hidden-features 32 \
  --out-features 8 \
  --grid-size 5 \
  --spline-order 3
```

Available options:

| Option | Default | Meaning |
| --- | --- | --- |
| `--quick` | off | Use short timings for smoke checks. |
| `--device` | `cpu` | Torch device, for example `cpu` or `cuda`. |
| `--dtype` | `float32` | Tensor dtype. Choices are `float32` and `float64`. |
| `--batch-size` | `128` | Number of rows in the benchmark input. |
| `--in-features` | `16` | Input feature count. |
| `--hidden-features` | `32` | Hidden width for the stacked `KAN` benchmark. |
| `--out-features` | `8` | Output feature count. |
| `--grid-size` | `5` | Number of spline grid intervals. |
| `--spline-order` | `3` | B-spline order. |

All dimension arguments must be positive. CUDA benchmarks require an available
CUDA device.

## What It Measures

The benchmark uses `torch.utils.benchmark` and reports timings for:

| Benchmark | Statement |
| --- | --- |
| `KANLinear.forward` | `layer(x)` |
| `KAN.forward` | `model(x)` for a two-layer `KAN([in, hidden, out])` |
| `KAN.forward backward` | forward, squared-mean loss, and backward pass |
| `KANLinear.update_grid` | `layer.update_grid(x)` |
| `KAN.regularization_loss` | `model.regularization_loss()` |

The command prints the benchmark context before the timings:

```text
efficient-kan benchmark
context: device=cpu, dtype=float32, batch_size=128, in_features=16, ...
```

Keep that context with any benchmark result you report.

## What It Does Not Measure

The benchmark does not measure:

- end-to-end model training throughput;
- data loading or preprocessing;
- optimiser step time;
- convergence speed;
- estimator accuracy, bias, variance, or coverage;
- memory peak usage;
- distributed or multi-GPU behaviour;
- performance across a full model-size grid;
- downstream simulation pipeline performance.

It also does not establish that one machine's timings predict another
machine's timings. PyTorch version, CPU, BLAS backend, CUDA version, GPU model,
driver, dtype, thread settings, and background load all matter.

## Responsible Same-Machine Comparisons

Use the benchmark to compare local changes under matched conditions.

A practical comparison protocol:

1. Start from a clean working tree or record the exact diff.
2. Record the current commit and package provenance:

   ```bash
   python scripts/provenance.py --json --source-checkout
   ```

3. Run the baseline benchmark with fixed options:

   ```bash
   python scripts/benchmark.py \
     --device cpu \
     --dtype float32 \
     --batch-size 256 \
     --in-features 32 \
     --hidden-features 64 \
     --out-features 8 \
     --grid-size 5 \
     --spline-order 3
   ```

4. Apply the candidate change.
5. Run the same command again on the same machine.
6. Compare medians and reported spread. Treat large interquartile-range
   warnings as evidence of environmental noise, not as a stable result.
7. Re-run both versions if the observed difference is small or surprising.

For Git-based comparisons, benchmark both commits with the same Python
environment and the same command. Do not compare a source checkout against an
unrelated installed package unless that is the intended comparison.

## CPU Notes

For CPU comparisons:

- close unrelated heavy workloads;
- use the same Python environment;
- use the same PyTorch build;
- use the same thread settings if you set variables such as
  `OMP_NUM_THREADS` or `MKL_NUM_THREADS`;
- prefer medians over individual run timings.

Example:

```bash
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
python scripts/benchmark.py --device cpu --dtype float32
```

Thread settings affect performance and should be recorded if used.

## CUDA Notes

For CUDA comparisons:

```bash
python scripts/benchmark.py --device cuda --dtype float32
```

CUDA timings can be affected by:

- GPU model and clock behaviour;
- driver and CUDA runtime;
- PyTorch CUDA build;
- other GPU processes;
- warmup effects;
- dtype and tensor shapes.

The benchmark command checks that CUDA is available when `--device cuda` is
requested, but it does not control GPU clocks or isolate the device from other
processes.

## Quick Mode

`--quick` is intended for validation gates and smoke checks:

```bash
python scripts/benchmark.py --quick
```

It uses shorter timing windows. A quick run is enough to catch gross failures
or obvious performance regressions, but it is not enough for careful
performance claims.

Use non-quick runs for comparisons that will be reported in issues, release
notes, or downstream research artefacts.

## Interpreting Results

`torch.utils.benchmark` may report warnings such as a large interquartile range.
Take those warnings seriously. They usually mean the timing environment is
noisy or the operation is too short relative to system fluctuation.

Good benchmark reporting includes:

- exact command;
- package commit and dirty status;
- Python version;
- PyTorch version;
- device and dtype;
- model dimensions;
- median timing and spread;
- whether the run used `--quick`.

Avoid reporting a single timing without context.

## Scope Boundary

This benchmark is for the core `efficient-kan` package. Parallel simulation
performance belongs in downstream statistical projects where the data-generating
process, estimator, replication structure, and artefact layout are defined.
