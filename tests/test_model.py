import pytest
import os
import pandas as pd
from sklearn.preprocessing import MinMaxScaler

torch = pytest.importorskip("torch")

from src.model_utils import (
    ShallowNN,
    NWRMSLELoss,
    TwoLayerNN,
    ResidualMLP,
    HybridLoss,
)  # adjust import


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


def test_hybrid_loss():
    loss_fn = HybridLoss(alpha=0.8)
    y_true = torch.tensor([[1.0, 3.0], [2.0, 4.0]])
    y_pred = torch.tensor([[1.5, 3.5], [2.5, 4.5]])
    w = torch.ones_like(y_true)

    loss = loss_fn(y_pred, y_true, w)
    assert loss >= 0, "Loss should be non-negative"

    # Test that alpha=1.0 gives pure NWRMSLE
    loss_fn_alpha1 = HybridLoss(alpha=1.0)
    loss_alpha1 = loss_fn_alpha1(y_pred, y_true, w)
    assert (
        pytest.approx(loss_alpha1.item(), abs=1e-6)
        == loss_fn.nwrmse(y_pred, y_true, w).item()
    )


def test_two_layer_nn_output_shape_and_reproducibility():
    torch.manual_seed(0)
    model = TwoLayerNN(input_dim=21)
    x = torch.randn(5, 21)

    y1 = model(x)
    y2 = model(x)

    assert y1.shape == (5, 21), "Expected output shape (batch, output_dim)"
    assert torch.allclose(
        y1, y2
    ), "Model forward pass should be deterministic given same seed"
    assert torch.all(y1 >= 0) and torch.all(
        y1 <= 1
    ), "Outputs should be clamped between 0 and 1"


def test_residual_mlp_output_shape_and_behavior():
    torch.manual_seed(0)
    model = ResidualMLP(input_dim=21, hidden=128, depth=3)
    x = torch.randn(5, 21)

    y = model(x)
    assert y.shape == (5, 21), "Expected output shape (batch, output_dim)"

    # Check that the output is close to the input due to residual connections
    # Note: We can't use exact equality due to the non-linear transformations
    #       but we should see some correlation between input and output
    assert torch.allclose(
        y.mean(dim=0), x.mean(dim=0), atol=0.5
    ), "Output should maintain some correlation with input due to residual connections"

    # Check that outputs are clamped between 0 and 1
    assert torch.all(y >= 0) and torch.all(
        y <= 1
    ), "Outputs should be clamped between 0 and 1"


def test_model_loading_and_prediction():
    # Create a temporary model
    model = TwoLayerNN(input_dim=10)
    model_path = "test_model.pth"
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "store_item": "test_store_item",
            "feature_cols": ["feat1", "feat2"],
        },
        model_path,
    )

    try:
        # Test loading
        loaded_store_item, loaded_model, loaded_feature_cols = load_model(model_path)
        assert loaded_store_item == "test_store_item"
        assert loaded_feature_cols == ["feat1", "feat2"]

        # Test prediction
        x = torch.randn(1, 10)
        pred = loaded_model(x)
        assert pred.shape == (1, 10)
        assert torch.all(pred >= 0) and torch.all(pred <= 1)
    finally:
        # Clean up
        if os.path.exists(model_path):
            os.remove(model_path)


def test_predict_next_days_for_sid():
    # Create mock data
    last_date_df = pd.DataFrame(
        {"store_item": ["test_store_item"], "feat1": [0.5], "feat2": [0.3]}
    )

    # Create mock model and scaler
    model = TwoLayerNN(input_dim=2)
    feature_cols = ["feat1", "feat2"]
    models = {"test_store_item": (model, feature_cols)}

    # Create mock scaler
    scaler = MinMaxScaler()
    scaler.fit(np.array([[0, 0], [1, 1]]))
    y_scalers = {"test_store_item": scaler}

    # Test prediction
    predictions = predict_next_days_for_sid(
        sid="test_store_item",
        last_date_df=last_date_df,
        models=models,
        y_scalers=y_scalers,
        days_to_predict=2,
    )

    assert len(predictions) == 2
    assert "store_item" in predictions.columns
    assert "date" in predictions.columns

    # Check that predictions are in the correct range (0-1)
    assert predictions.iloc[:, 2:].min().min() >= 0
    assert predictions.iloc[:, 2:].max().max() <= 1
