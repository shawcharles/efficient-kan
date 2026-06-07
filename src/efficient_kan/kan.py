import math
from collections.abc import Sequence
from numbers import Real

import torch
import torch.nn.functional as F


def _validate_finite_scalar(name: str, value: float) -> float:
    if not isinstance(value, Real) or isinstance(value, bool) or not math.isfinite(value):
        raise ValueError(f"{name} must be a finite scalar")
    return float(value)


def _validate_grid_range(grid_range: Sequence[float]) -> tuple[float, float]:
    try:
        lower_raw, upper_raw = grid_range
    except (TypeError, ValueError):
        raise ValueError("grid_range must contain two increasing finite values")
    lower = _validate_finite_scalar("grid_range[0]", lower_raw)
    upper = _validate_finite_scalar("grid_range[1]", upper_raw)
    if lower >= upper:
        raise ValueError("grid_range must contain two increasing finite values")
    return lower, upper


class KANLinear(torch.nn.Module):
    """KAN layer with a base linear path and an adaptive B-spline path.

    Inputs must have last dimension ``in_features``. ``forward()`` accepts any
    number of leading dimensions and restores them on output. ``update_grid()``
    follows the same leading-dimension convention, mutates the spline grid, and
    refits spline coefficients to preserve the layer's spline contribution at
    the supplied sample points.
    """

    def __init__(
        self,
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
    ) -> None:
        super().__init__()
        if in_features <= 0 or out_features <= 0:
            raise ValueError("in_features and out_features must be positive")
        if grid_size <= 0:
            raise ValueError("grid_size must be positive")
        if spline_order <= 0:
            raise ValueError("spline_order must be positive")
        scale_noise = _validate_finite_scalar("scale_noise", scale_noise)
        scale_base = _validate_finite_scalar("scale_base", scale_base)
        scale_spline = _validate_finite_scalar("scale_spline", scale_spline)
        grid_eps = _validate_finite_scalar("grid_eps", grid_eps)
        if not 0.0 <= grid_eps <= 1.0:
            raise ValueError("grid_eps must be in the interval [0, 1]")
        grid_lower, grid_upper = _validate_grid_range(grid_range)

        self.in_features = in_features
        self.out_features = out_features
        self.grid_size = grid_size
        self.spline_order = spline_order

        h = (grid_upper - grid_lower) / grid_size
        grid = (
            (
                torch.arange(-spline_order, grid_size + spline_order + 1) * h
                + grid_lower
            )
            .expand(in_features, -1)
            .contiguous()
        )
        self.register_buffer("grid", grid)

        self.base_weight = torch.nn.Parameter(torch.Tensor(out_features, in_features))
        self.spline_weight = torch.nn.Parameter(
            torch.Tensor(out_features, in_features, grid_size + spline_order)
        )
        if enable_standalone_scale_spline:
            self.spline_scaler = torch.nn.Parameter(
                torch.Tensor(out_features, in_features)
            )

        self.scale_noise = scale_noise
        self.scale_base = scale_base
        self.scale_spline = scale_spline
        self.enable_standalone_scale_spline = enable_standalone_scale_spline
        self.base_activation = base_activation()
        self.grid_eps = grid_eps

        self.reset_parameters()

    def reset_parameters(self) -> None:
        torch.nn.init.kaiming_uniform_(self.base_weight, a=math.sqrt(5))
        with torch.no_grad():
            self.base_weight.mul_(self.scale_base)
            noise = (
                (
                    torch.rand(self.grid_size + 1, self.in_features, self.out_features)
                    - 1 / 2
                )
                * self.scale_noise
                / self.grid_size
            )
            self.spline_weight.copy_(
                (self.scale_spline if not self.enable_standalone_scale_spline else 1.0)
                * self.curve2coeff(
                    self.grid.T[self.spline_order : -self.spline_order],
                    noise,
                )
            )
            if self.enable_standalone_scale_spline:
                # torch.nn.init.constant_(self.spline_scaler, self.scale_spline)
                torch.nn.init.kaiming_uniform_(self.spline_scaler, a=math.sqrt(5))
                self.spline_scaler.mul_(self.scale_spline)

    def _validate_input(self, x: torch.Tensor, *, name: str = "x") -> None:
        if x.dim() != 2 or x.size(1) != self.in_features:
            raise ValueError(
                f"{name} must have shape (batch_size, {self.in_features}); "
                f"got {tuple(x.shape)}"
            )
        if x.size(0) == 0:
            raise ValueError(f"{name} must contain at least one sample")
        if not torch.isfinite(x).all():
            raise ValueError(f"{name} must contain only finite values")

    def _flatten_feature_tensor(self, x: torch.Tensor, *, name: str = "x") -> torch.Tensor:
        if x.dim() == 0 or x.size(-1) != self.in_features:
            raise ValueError(
                f"{name} must have last dimension {self.in_features}; got {tuple(x.shape)}"
            )
        return x.reshape(-1, self.in_features)

    def _b_splines(self, x: torch.Tensor, grid: torch.Tensor) -> torch.Tensor:
        x = x.unsqueeze(-1)
        bases = ((x >= grid[:, :-1]) & (x < grid[:, 1:])).to(x.dtype)
        for k in range(1, self.spline_order + 1):
            bases = (
                (x - grid[:, : -(k + 1)])
                / (grid[:, k:-1] - grid[:, : -(k + 1)])
                * bases[:, :, :-1]
            ) + (
                (grid[:, k + 1 :] - x)
                / (grid[:, k + 1 :] - grid[:, 1:(-k)])
                * bases[:, :, 1:]
            )

        expected_shape = (
            x.size(0),
            self.in_features,
            self.grid_size + self.spline_order,
        )
        if bases.size() != expected_shape:
            raise RuntimeError(f"Unexpected B-spline basis shape: {tuple(bases.shape)}")
        return bases.contiguous()

    def b_splines(self, x: torch.Tensor) -> torch.Tensor:
        """Compute B-spline basis values on the current grid.

        Args:
            x: Finite tensor of shape ``(batch_size, in_features)``.

        Returns:
            Tensor of shape ``(batch_size, in_features, grid_size + spline_order)``.
            For in-range points and strictly increasing grids, basis values are
            non-negative and form a partition of unity up to floating-point
            tolerance.
        """
        self._validate_input(x)

        grid: torch.Tensor = (
            self.grid
        )  # (in_features, grid_size + 2 * spline_order + 1)
        return self._b_splines(x, grid)

    def _curve2coeff_with_grid(
        self,
        x: torch.Tensor,
        y: torch.Tensor,
        grid: torch.Tensor,
    ) -> torch.Tensor:
        A = self._b_splines(x, grid).transpose(
            0, 1
        )  # (in_features, batch_size, grid_size + spline_order)
        B = y.transpose(0, 1)  # (in_features, batch_size, out_features)
        try:
            solution = torch.linalg.lstsq(
                A, B
            ).solution  # (in_features, grid_size + spline_order, out_features)
        except RuntimeError as exc:
            raise RuntimeError(
                "Failed to solve spline interpolation coefficients; "
                "check that inputs are finite and the spline basis is well-conditioned."
            ) from exc
        result = solution.permute(
            2, 0, 1
        )  # (out_features, in_features, grid_size + spline_order)

        expected_result_shape = (
            self.out_features,
            self.in_features,
            self.grid_size + self.spline_order,
        )
        if result.size() != expected_result_shape:
            raise RuntimeError(
                f"Unexpected coefficient shape: {tuple(result.shape)}"
            )
        return result.contiguous()

    def curve2coeff(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """Solve B-spline coefficients for sample values by least squares.

        Args:
            x: Finite tensor of shape ``(batch_size, in_features)``.
            y: Finite tensor of shape ``(batch_size, in_features, out_features)``.

        Returns:
            Coefficient tensor of shape
            ``(out_features, in_features, grid_size + spline_order)``. The solve
            interpolates when the spline basis has enough rank for the supplied
            points; otherwise it returns the least-squares/minimum-norm solution
            produced by ``torch.linalg.lstsq``.
        """
        self._validate_input(x)
        expected_y_shape = (x.size(0), self.in_features, self.out_features)
        if y.size() != expected_y_shape:
            raise ValueError(f"y must have shape {expected_y_shape}; got {tuple(y.shape)}")
        if not torch.isfinite(y).all():
            raise ValueError("y must contain only finite values")

        return self._curve2coeff_with_grid(x, y, self.grid)

    @property
    def scaled_spline_weight(self) -> torch.Tensor:
        """Spline coefficients after applying the optional standalone scaler."""
        return self.spline_weight * (
            self.spline_scaler.unsqueeze(-1)
            if self.enable_standalone_scale_spline
            else 1.0
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Evaluate the layer for inputs with last dimension ``in_features``."""
        if x.dim() == 0 or x.size(-1) != self.in_features:
            got = x.size(-1) if x.dim() > 0 else tuple(x.shape)
            raise ValueError(f"Expected input last dimension {self.in_features}; got {got}")
        original_shape = x.shape
        x = x.reshape(-1, self.in_features)

        base_output = F.linear(self.base_activation(x), self.base_weight)
        spline_output = F.linear(
            self.b_splines(x).view(x.size(0), -1),
            self.scaled_spline_weight.view(self.out_features, -1),
        )
        output = base_output + spline_output

        output = output.reshape(*original_shape[:-1], self.out_features)
        return output

    @torch.no_grad()
    def update_grid(self, x: torch.Tensor, margin: float = 0.01) -> None:
        """Adapt the spline grid to ``x`` and refit spline coefficients.

        ``x`` follows the same shape convention as ``forward()``: any leading
        dimensions are flattened into the sample dimension. The method mutates
        ``grid`` and ``spline_weight`` in place. Coefficients are refit so the
        spline contribution at the supplied samples is preserved up to the
        accuracy of the least-squares solve.
        """
        margin = _validate_finite_scalar("margin", margin)
        if margin <= 0:
            raise ValueError("margin must be positive")
        x = self._flatten_feature_tensor(x)
        self._validate_input(x)
        batch = x.size(0)

        splines = self.b_splines(x)  # (batch, in, coeff)
        splines = splines.permute(1, 0, 2)  # (in, batch, coeff)
        orig_coeff = self.scaled_spline_weight  # (out, in, coeff)
        orig_coeff = orig_coeff.permute(1, 2, 0)  # (in, coeff, out)
        unreduced_spline_output = torch.bmm(splines, orig_coeff)  # (in, batch, out)
        unreduced_spline_output = unreduced_spline_output.permute(
            1, 0, 2
        )  # (batch, in, out)
        if not torch.isfinite(unreduced_spline_output).all():
            raise ValueError("Spline outputs must be finite before grid update")

        # sort each channel individually to collect data distribution
        x_sorted = torch.sort(x, dim=0)[0]
        grid_adaptive = x_sorted[
            torch.linspace(
                0, batch - 1, self.grid_size + 1, dtype=torch.int64, device=x.device
            )
        ]

        uniform_step = (x_sorted[-1] - x_sorted[0] + 2 * margin) / self.grid_size
        grid_uniform = (
            torch.arange(
                self.grid_size + 1, dtype=x.dtype, device=x.device
            ).unsqueeze(1)
            * uniform_step
            + x_sorted[0]
            - margin
        )

        grid = self.grid_eps * grid_uniform + (1 - self.grid_eps) * grid_adaptive
        grid = torch.concatenate(
            [
                grid[:1]
                - uniform_step
                * torch.arange(self.spline_order, 0, -1, device=x.device).unsqueeze(1),
                grid,
                grid[-1:]
                + uniform_step
                * torch.arange(1, self.spline_order + 1, device=x.device).unsqueeze(1),
            ],
            dim=0,
        )
        if not torch.all(grid[1:] > grid[:-1]):
            raise RuntimeError("Updated grid must be strictly increasing")

        proposed_grid = grid.T.contiguous()
        new_scaled_spline_weight = self._curve2coeff_with_grid(
            x,
            unreduced_spline_output,
            proposed_grid,
        )
        if self.enable_standalone_scale_spline:
            spline_scaler = self.spline_scaler.unsqueeze(-1)
            safe_scaler = torch.where(
                spline_scaler == 0,
                torch.ones_like(spline_scaler),
                spline_scaler,
            )
            new_spline_weight = new_scaled_spline_weight / safe_scaler
            new_spline_weight = torch.where(
                spline_scaler == 0,
                torch.zeros_like(new_spline_weight),
                new_spline_weight,
            )
        else:
            new_spline_weight = new_scaled_spline_weight
        self.grid.copy_(proposed_grid)
        self.spline_weight.copy_(new_spline_weight)

    def regularization_loss(
        self,
        regularize_activation: float = 1.0,
        regularize_entropy: float = 1.0,
    ) -> torch.Tensor:
        """Return the efficient spline-weight regularization penalty.

        The activation term is the sum of mean absolute spline-weight magnitudes
        over output/input feature pairs. The optional entropy term is computed
        from those normalized magnitudes. This preserves the memory-efficient
        implementation and is not identical to regularizing expanded per-sample
        spline activations.
        """
        l1_fake = self.spline_weight.abs().mean(-1)
        regularization_loss_activation = l1_fake.sum()
        regularization_loss_entropy = regularization_loss_activation.new_zeros(())
        if regularize_entropy != 0 and regularization_loss_activation > 0:
            p = l1_fake / regularization_loss_activation
            p = p[p > 0]
            regularization_loss_entropy = -torch.sum(p * p.log())
        return (
            regularize_activation * regularization_loss_activation
            + regularize_entropy * regularization_loss_entropy
        )


class KAN(torch.nn.Module):
    """Stack of ``KANLinear`` layers.

    ``layers_hidden`` contains the input width, any hidden widths, and the output
    width. ``forward(update_grid=True)`` updates each layer's adaptive grid using
    the current layer input before evaluating that layer; this mutates model
    state and accepts the same leading-dimension input convention as normal
    ``forward()``.
    """

    def __init__(
        self,
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
    ) -> None:
        super().__init__()
        if len(layers_hidden) < 2:
            raise ValueError("layers_hidden must contain at least input and output sizes")
        self.grid_size = grid_size
        self.spline_order = spline_order

        self.layers = torch.nn.ModuleList()
        for in_features, out_features in zip(layers_hidden, layers_hidden[1:]):
            self.layers.append(
                KANLinear(
                    in_features,
                    out_features,
                    grid_size=grid_size,
                    spline_order=spline_order,
                    scale_noise=scale_noise,
                    scale_base=scale_base,
                    scale_spline=scale_spline,
                    enable_standalone_scale_spline=enable_standalone_scale_spline,
                    base_activation=base_activation,
                    grid_eps=grid_eps,
                    grid_range=grid_range,
                )
            )

    def forward(self, x: torch.Tensor, update_grid: bool = False) -> torch.Tensor:
        """Evaluate the network, optionally adapting each layer grid in place."""
        for layer in self.layers:
            if update_grid:
                layer.update_grid(x)
            x = layer(x)
        return x

    def regularization_loss(
        self,
        regularize_activation: float = 1.0,
        regularize_entropy: float = 1.0,
    ) -> torch.Tensor:
        """Sum regularization penalties across all KAN layers."""
        return sum(
            layer.regularization_loss(regularize_activation, regularize_entropy)
            for layer in self.layers
        )
