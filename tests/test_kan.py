import torch
import torch.nn as nn
import torch.nn.functional as F
import pytest

from efficient_kan import KAN, KANLinear, __version__


def test_package_exposes_release_version():
    assert __version__ == "0.2.0"


def test_kan_linear_forward_shape_for_2d_input():
    torch.manual_seed(0)
    layer = KANLinear(3, 2)

    output = layer(torch.randn(7, 3))

    assert output.shape == (7, 2)
    assert torch.isfinite(output).all()


def test_kan_forward_preserves_leading_dimensions():
    torch.manual_seed(0)
    model = KAN([3, 4, 2])

    output = model(torch.randn(5, 6, 3))

    assert output.shape == (5, 6, 2)
    assert torch.isfinite(output).all()


def test_kan_forward_with_grid_update_preserves_leading_dimensions():
    torch.manual_seed(0)
    model = KAN([3, 4, 2])
    x = torch.randn(5, 6, 3)

    output = model(x, update_grid=True)

    assert output.shape == (5, 6, 2)
    assert torch.isfinite(output).all()


def test_forward_rejects_bad_input_dimension():
    model = KAN([3, 2])

    with pytest.raises(ValueError, match="Expected input last dimension"):
        model(torch.randn(4, 2))


@pytest.mark.parametrize(
    "kwargs,match",
    [
        ({"in_features": 0}, "in_features and out_features must be positive"),
        ({"out_features": 0}, "in_features and out_features must be positive"),
        ({"grid_size": 0}, "grid_size must be positive"),
        ({"spline_order": 0}, "spline_order must be positive"),
        ({"grid_range": (1.0, -1.0)}, "grid_range"),
        ({"grid_eps": -0.1}, "grid_eps must be in the interval"),
        ({"grid_eps": 1.1}, "grid_eps must be in the interval"),
        ({"grid_eps": float("nan")}, "grid_eps must be a finite scalar"),
        ({"scale_noise": float("inf")}, "scale_noise must be a finite scalar"),
        ({"scale_base": float("-inf")}, "scale_base must be a finite scalar"),
        ({"scale_spline": float("nan")}, "scale_spline must be a finite scalar"),
    ],
)
def test_kan_linear_rejects_invalid_constructor_arguments(kwargs, match):
    params = {
        "in_features": 2,
        "out_features": 3,
        "grid_size": 4,
        "spline_order": 2,
        "grid_range": (-1.0, 1.0),
    }
    params.update(kwargs)

    with pytest.raises(ValueError, match=match):
        KANLinear(**params)


def test_b_splines_are_finite_and_have_expected_shape():
    torch.manual_seed(0)
    layer = KANLinear(2, 3, grid_size=4, spline_order=2)

    bases = layer.b_splines(torch.linspace(-1, 1, 10).reshape(5, 2))

    assert bases.shape == (5, 2, 6)
    assert torch.isfinite(bases).all()


def test_b_splines_are_nonnegative_partition_of_unity_on_interior_points():
    torch.manual_seed(0)
    layer = KANLinear(2, 3, grid_size=5, spline_order=3)
    x = torch.linspace(-0.9, 0.9, 24).reshape(12, 2)

    bases = layer.b_splines(x)

    assert torch.all(bases >= -1e-7)
    assert torch.allclose(bases.sum(dim=-1), torch.ones_like(x), atol=1e-6)


def test_curve2coeff_returns_expected_shape_and_finite_values():
    torch.manual_seed(0)
    layer = KANLinear(2, 3, grid_size=4, spline_order=2)
    x = torch.linspace(-1, 1, 10).reshape(5, 2)
    y = torch.randn(5, 2, 3)

    coeff = layer.curve2coeff(x, y)

    assert coeff.shape == (3, 2, 6)
    assert torch.isfinite(coeff).all()


def test_curve2coeff_reconstructs_training_values():
    torch.manual_seed(0)
    layer = KANLinear(2, 3, grid_size=4, spline_order=2)
    x = torch.linspace(-0.8, 0.8, 12).reshape(6, 2)
    y = torch.randn(6, 2, 3)

    coeff = layer.curve2coeff(x, y)
    reconstruction = torch.einsum("bic,oic->bio", layer.b_splines(x), coeff)

    assert torch.allclose(reconstruction, y, atol=2e-5, rtol=2e-5)


def test_update_grid_keeps_outputs_finite():
    torch.manual_seed(0)
    layer = KANLinear(2, 3)
    x = torch.rand(64, 2)

    for _ in range(3):
        layer.update_grid(x)
        output = layer(x)
        assert torch.isfinite(layer.grid).all()
        assert torch.isfinite(layer.spline_weight).all()
        assert torch.isfinite(output).all()


def test_update_grid_keeps_grid_strictly_increasing():
    torch.manual_seed(0)
    layer = KANLinear(2, 3)
    x = torch.rand(64, 2)

    layer.update_grid(x)

    assert torch.all(layer.grid[:, 1:] > layer.grid[:, :-1])


def test_update_grid_rejects_degenerate_pure_adaptive_grid_without_mutation():
    layer = KANLinear(2, 3, grid_eps=0.0)
    old_grid = layer.grid.detach().clone()
    old_spline_weight = layer.spline_weight.detach().clone()

    with pytest.raises(RuntimeError, match="strictly increasing"):
        layer.update_grid(torch.zeros(16, 2))

    assert torch.equal(layer.grid, old_grid)
    assert torch.equal(layer.spline_weight, old_spline_weight)


@pytest.mark.parametrize("enable_standalone_scale_spline", [False, True])
def test_update_grid_preserves_spline_output_at_samples(enable_standalone_scale_spline):
    torch.manual_seed(0)
    layer = KANLinear(
        2,
        3,
        grid_size=5,
        spline_order=3,
        enable_standalone_scale_spline=enable_standalone_scale_spline,
    )
    if enable_standalone_scale_spline:
        with torch.no_grad():
            layer.spline_scaler.fill_(2.0)
    x = torch.linspace(-0.8, 0.8, 16).reshape(8, 2)

    before = F.linear(
        layer.b_splines(x).view(x.size(0), -1),
        layer.scaled_spline_weight.view(layer.out_features, -1),
    )
    layer.update_grid(x, margin=0.05)
    after = F.linear(
        layer.b_splines(x).view(x.size(0), -1),
        layer.scaled_spline_weight.view(layer.out_features, -1),
    )

    assert torch.allclose(after, before, atol=2e-5, rtol=2e-5)


def test_update_grid_preserves_zero_standalone_spline_scaler():
    torch.manual_seed(0)
    layer = KANLinear(2, 3, scale_spline=0.0)
    x = torch.linspace(-0.8, 0.8, 16).reshape(8, 2)

    before = F.linear(
        layer.b_splines(x).view(x.size(0), -1),
        layer.scaled_spline_weight.view(layer.out_features, -1),
    )
    layer.update_grid(x, margin=0.05)
    after = F.linear(
        layer.b_splines(x).view(x.size(0), -1),
        layer.scaled_spline_weight.view(layer.out_features, -1),
    )

    assert torch.allclose(before, torch.zeros_like(before))
    assert torch.allclose(after, before)
    assert torch.isfinite(layer.spline_weight).all()


@pytest.mark.parametrize("margin", [0.0, -0.01, float("nan")])
def test_update_grid_rejects_invalid_margin(margin):
    layer = KANLinear(2, 3)

    with pytest.raises(ValueError):
        layer.update_grid(torch.rand(8, 2), margin=margin)


def test_b_splines_rejects_invalid_x():
    layer = KANLinear(2, 3)

    with pytest.raises(ValueError, match="x must have shape"):
        layer.b_splines(torch.rand(8, 3))
    with pytest.raises(ValueError, match="x must contain only finite values"):
        layer.b_splines(torch.tensor([[0.0, float("nan")]]))


def test_curve2coeff_rejects_invalid_y():
    layer = KANLinear(2, 3)
    x = torch.rand(8, 2)

    with pytest.raises(ValueError, match="y must have shape"):
        layer.curve2coeff(x, torch.rand(8, 2, 2))
    with pytest.raises(ValueError, match="y must contain only finite values"):
        layer.curve2coeff(x, torch.full((8, 2, 3), float("inf")))


def test_cpu_double_precision_forward_backward_update_grid_smoke():
    torch.manual_seed(0)
    model = KAN([2, 3, 1]).double()
    x = torch.linspace(-0.5, 0.5, 16, dtype=torch.float64).reshape(8, 2)

    output = model(x, update_grid=True)
    loss = output.square().mean() + 1e-5 * model.regularization_loss(1.0, 0.0)
    loss.backward()

    assert output.dtype == torch.float64
    assert torch.isfinite(output).all()
    for parameter in model.parameters():
        assert parameter.grad is None or torch.isfinite(parameter.grad).all()


def test_regularization_loss_is_finite_when_spline_weights_are_zero():
    layer = KANLinear(2, 3)
    with torch.no_grad():
        layer.spline_weight.zero_()

    loss = layer.regularization_loss()
    activation_only_loss = layer.regularization_loss(regularize_activation=1.0, regularize_entropy=0.0)

    assert torch.isfinite(loss)
    assert torch.isfinite(activation_only_loss)
    assert loss.item() == 0.0
    assert activation_only_loss.item() == 0.0


def test_regularization_loss_backpropagates_finite_gradients():
    torch.manual_seed(0)
    layer = KANLinear(2, 3)

    loss = layer.regularization_loss()
    loss.backward()

    assert layer.spline_weight.grad is not None
    assert torch.isfinite(layer.spline_weight.grad).all()


def test_scale_base_increases_base_weight_magnitude():
    torch.manual_seed(123)
    small = KANLinear(100, 1, scale_base=0.5)
    torch.manual_seed(123)
    large = KANLinear(100, 1, scale_base=2.0)

    assert large.base_weight.detach().std() > small.base_weight.detach().std()


def test_kan_exposes_standalone_spline_scale_option():
    model = KAN([2, 3, 1], enable_standalone_scale_spline=False)

    assert all(not layer.enable_standalone_scale_spline for layer in model.layers)


def test_state_dict_round_trip_preserves_outputs():
    torch.manual_seed(0)
    model = KAN([2, 3, 1])
    clone = KAN([2, 3, 1])
    x = torch.randn(8, 2)

    clone.load_state_dict(model.state_dict())

    assert torch.allclose(model(x), clone(x))


def test_tiny_training_smoke_reduces_loss():
    torch.manual_seed(0)
    model = KAN([2, 4, 1], base_activation=nn.Identity)
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.03, weight_decay=1e-4)
    x = torch.rand(128, 2)
    target = ((x[:, :1] + x[:, 1:]) / (1 + x[:, :1] * x[:, 1:])).detach()

    with torch.no_grad():
        initial_loss = F.mse_loss(model(x), target)

    for _ in range(30):
        optimizer.zero_grad()
        loss = F.mse_loss(model(x), target) + 1e-5 * model.regularization_loss(1.0, 0.0)
        loss.backward()
        optimizer.step()

    with torch.no_grad():
        final_loss = F.mse_loss(model(x), target)

    assert torch.isfinite(final_loss)
    assert final_loss < initial_loss
