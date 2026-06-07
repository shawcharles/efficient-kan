"""Local microbenchmarks for efficient-kan."""

from __future__ import annotations

import argparse
from typing import Sequence

import torch
from torch.utils import benchmark

from efficient_kan import KAN, KANLinear


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run local efficient-kan benchmarks.")
    parser.add_argument("--quick", action="store_true", help="Use short timings for smoke checks.")
    parser.add_argument("--device", default="cpu", help="Torch device, for example cpu or cuda.")
    parser.add_argument("--dtype", choices=("float32", "float64"), default="float32")
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--in-features", type=int, default=16)
    parser.add_argument("--hidden-features", type=int, default=32)
    parser.add_argument("--out-features", type=int, default=8)
    parser.add_argument("--grid-size", type=int, default=5)
    parser.add_argument("--spline-order", type=int, default=3)
    return parser.parse_args(argv)


def _timer(name: str, stmt: str, globals_: dict[str, object], min_run_time: float) -> None:
    result = benchmark.Timer(
        stmt=stmt,
        globals=globals_,
        label="efficient-kan",
        sub_label=name,
    ).blocked_autorange(min_run_time=min_run_time)
    print(result)


def main(argv: Sequence[str] | None = None) -> None:
    args = _parse_args(argv)
    if args.device == "cuda" and not torch.cuda.is_available():
        raise SystemExit("CUDA was requested but is not available.")
    if min(
        args.batch_size,
        args.in_features,
        args.hidden_features,
        args.out_features,
        args.grid_size,
        args.spline_order,
    ) <= 0:
        raise SystemExit("Benchmark dimensions must be positive.")

    dtype = getattr(torch, args.dtype)
    device = torch.device(args.device)
    min_run_time = 0.05 if args.quick else 0.5
    torch.manual_seed(0)

    layer = KANLinear(
        args.in_features,
        args.out_features,
        grid_size=args.grid_size,
        spline_order=args.spline_order,
    ).to(device=device, dtype=dtype)
    model = KAN(
        [args.in_features, args.hidden_features, args.out_features],
        grid_size=args.grid_size,
        spline_order=args.spline_order,
    ).to(device=device, dtype=dtype)
    x = torch.randn(args.batch_size, args.in_features, device=device, dtype=dtype)
    x_backward = x.detach().clone().requires_grad_(True)

    def run_backward() -> None:
        model.zero_grad(set_to_none=True)
        if x_backward.grad is not None:
            x_backward.grad = None
        model(x_backward).square().mean().backward()

    print("efficient-kan benchmark")
    print(
        "context: "
        f"device={device}, dtype={args.dtype}, batch_size={args.batch_size}, "
        f"in_features={args.in_features}, hidden_features={args.hidden_features}, "
        f"out_features={args.out_features}, grid_size={args.grid_size}, "
        f"spline_order={args.spline_order}, quick={args.quick}"
    )

    timer_globals: dict[str, object] = {
        "layer": layer,
        "model": model,
        "x": x,
        "run_backward": run_backward,
    }
    _timer("KANLinear.forward", "layer(x)", timer_globals, min_run_time)
    _timer("KAN.forward", "model(x)", timer_globals, min_run_time)
    _timer("KAN.forward backward", "run_backward()", timer_globals, min_run_time)
    _timer("KANLinear.update_grid", "layer.update_grid(x)", timer_globals, min_run_time)
    _timer("KAN.regularization_loss", "model.regularization_loss()", timer_globals, min_run_time)


if __name__ == "__main__":
    main()
