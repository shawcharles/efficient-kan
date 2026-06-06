import torch
import torch.nn as nn
import torch.nn.functional as F
import pytest

from efficient_kan import KAN, KANLinear


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
        ({"grid_range": (1.0, -1.0)}, "grid_range must contain two increasing values"),
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


def test_curve2coeff_returns_expected_shape_and_finite_values():
    torch.manual_seed(0)
    layer = KANLinear(2, 3, grid_size=4, spline_order=2)
    x = torch.linspace(-1, 1, 10).reshape(5, 2)
    y = torch.randn(5, 2, 3)

    coeff = layer.curve2coeff(x, y)

    assert coeff.shape == (3, 2, 6)
    assert torch.isfinite(coeff).all()


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
