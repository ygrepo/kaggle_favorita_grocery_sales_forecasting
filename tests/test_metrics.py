import logging
import pytest


torch = pytest.importorskip("torch")

from src.model import MeanAbsoluteErrorLog1p, compute_mae


def test_mean_absolute_error_log1p_and_compute_mae_agree():
    # Original-unit targets and predictions
    # Shape: (batch=2, dims=3)
    Y = torch.tensor([[0.0, 1.0, 10.0], [5.0, 0.0, 2.0]], dtype=torch.float32)
    P = torch.tensor([[0.0, 2.0, 11.0], [4.0, 1.0, 2.5]], dtype=torch.float32)

    # Convert to log1p space (as expected by both implementations)
    yb = torch.log1p(Y)
    preds = torch.log1p(P)

    # Only evaluate on columns 1 and 2
    sales_idx = [1, 2]

    # Metric class
    metric = MeanAbsoluteErrorLog1p(sales_idx=sales_idx)
    metric.update(preds, yb)
    mae_metric = metric.compute().item()

    # Functional version
    logger = logging.getLogger("test_compute_mae")
    mae_fn = compute_mae(preds, yb, sales_idx, logger)

    # Expected manual computation in original units over target > 0 mask:
    # Included diffs: |2-1|=1, |11-10|=1, |2.5-2|=0.5 (note: target==0 excluded) -> mean = 2.5/3 ≈ 0.8333333
    expected = 2.5 / 3.0
    assert pytest.approx(mae_metric, rel=1e-6, abs=1e-6) == expected
    assert pytest.approx(mae_fn, rel=1e-6, abs=1e-6) == expected
    # Ensure both implementations agree closely
    assert pytest.approx(mae_metric, rel=1e-6, abs=1e-6) == mae_fn


def test_mean_absolute_error_log1p_zero_targets_yield_zero():
    # For selected indices, targets are all zero -> mask has no positives
    Y = torch.zeros((3, 2), dtype=torch.float32)
    P = torch.tensor([[1.0, 5.0], [2.0, 3.0], [10.0, 0.0]], dtype=torch.float32)

    yb = torch.log1p(Y)
    preds = torch.log1p(P)

    sales_idx = [0, 1]

    metric = MeanAbsoluteErrorLog1p(sales_idx=sales_idx)
    metric.update(preds, yb)
    mae_metric = metric.compute().item()

    logger = logging.getLogger("test_compute_mae_zero")
    mae_fn = compute_mae(preds, yb, sales_idx, logger)

    assert mae_metric == 0.0
    assert mae_fn == 0.0
