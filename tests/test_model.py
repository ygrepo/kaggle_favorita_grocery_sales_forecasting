import torch
import pytest

from src.model_utils import ShallowNN, NWRMSLELoss  # adjust import


def test_shallow_nn_output_shape_and_reproducibility():
    torch.manual_seed(0)
    model = ShallowNN(input_dim=21, hidden_dim=10)
    # create a batch of 5 samples, each with 21 features
    x = torch.randn(5, 21)

    # first forward
    y1 = model(x)
    # second forward (same weights & input)
    y2 = model(x)

    # check shape
    assert y1.shape == (5, 21), "Expected output shape (batch, output_dim)"
    # check determinism
    assert torch.allclose(
        y1, y2
    ), "Model forward pass should be deterministic given same seed"


def test_nwrmsle_loss_zero_when_preds_equal_targets():
    loss_fn = NWRMSLELoss()
    # any positive targets
    y_true = torch.tensor([[0.0, 1.0], [2.0, 3.0]])
    y_pred = y_true.clone()
    # uniform weights
    w = torch.ones_like(y_true)

    loss = loss_fn(y_pred, y_true, w)
    assert (
        pytest.approx(0.0, abs=1e-6) == loss.item()
    ), "Loss should be zero when predictions == targets"


def test_nwrmsle_loss_known_values():
    loss_fn = NWRMSLELoss()
    # single-sample, two dims
    y_true = torch.tensor([[1.0, 3.0]])
    y_pred = torch.tensor([[2.0, 6.0]])
    # weights
    w = torch.tensor([[0.5, 1.5]])

    # compute by hand:
    # ln(2+1)-ln(1+1) = ln3 - ln2
    # ln(6+1)-ln(3+1) = ln7 - ln4
    import math

    d1 = math.log(3) - math.log(2)
    d2 = math.log(7) - math.log(4)
    num = 0.5 * d1**2 + 1.5 * d2**2
    den = 0.5 + 1.5
    expected = math.sqrt(num / den)

    loss = loss_fn(y_pred, y_true, w)
    assert (
        pytest.approx(expected, rel=1e-6) == loss.item()
    ), "Loss should match hand‑computed NWRMSLE"
