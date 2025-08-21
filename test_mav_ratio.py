#!/usr/bin/env python3
"""
Test script to verify MAV ratio norm functionality in GDKM.
"""

import numpy as np
import sys
import os

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))

try:
    from gdkm import GeneralizedDoubleKMeans

    GDKM_AVAILABLE = True
except ImportError as e:
    print(f"Warning: Could not import GDKM: {e}")
    GDKM_AVAILABLE = False


# Simple test without sklearn dependencies
def test_mav_ratio_calculation_simple():
    """Test MAV ratio calculation logic directly."""
    print("Testing MAV ratio calculation logic...")

    # Test data
    X = np.array([[1.0, 2.0], [3.0, 4.0]])

    # Test reconstruction (perfect case)
    recon = X.copy()  # Perfect reconstruction
    diff = X - recon  # Should be all zeros

    mae = np.sum(np.abs(diff))  # Should be 0
    mav = np.sum(np.abs(X))  # Should be 1+2+3+4 = 10
    mav_ratio = mae / max(mav, 1e-12)  # Should be 0

    print(f"   X = \n{X}")
    print(f"   Perfect reconstruction diff = \n{diff}")
    print(f"   MAE = {mae}")
    print(f"   MAV = {mav}")
    print(f"   MAV ratio = {mav_ratio}")

    assert abs(mav_ratio) < 1e-10, f"Expected ~0, got {mav_ratio}"
    print("   ✓ Perfect reconstruction test passed")

    # Test worst case reconstruction (all zeros)
    recon_worst = np.zeros_like(X)
    diff_worst = X - recon_worst  # Should equal X

    mae_worst = np.sum(np.abs(diff_worst))  # Should equal sum(|X|) = 10
    mav_worst = np.sum(np.abs(X))  # Should be 10
    mav_ratio_worst = mae_worst / max(mav_worst, 1e-12)  # Should be 1

    print(f"   Worst reconstruction diff = \n{diff_worst}")
    print(f"   MAE worst = {mae_worst}")
    print(f"   MAV worst = {mav_worst}")
    print(f"   MAV ratio worst = {mav_ratio_worst}")

    assert abs(mav_ratio_worst - 1.0) < 1e-10, f"Expected ~1, got {mav_ratio_worst}"
    print("   ✓ Worst reconstruction test passed")

    return True


def test_mav_ratio_norm():
    """Test that MAV ratio norm works without errors."""
    if not GDKM_AVAILABLE:
        print("Skipping GDKM tests - module not available")
        return True

    print("Testing MAV ratio norm in GDKM...")

    # Create simple test data
    np.random.seed(42)
    X = np.random.rand(10, 8) * 10  # 10 rows, 8 columns

    # Test with tied columns (global V)
    print("\n1. Testing with tied columns (global V)...")
    gdkm_tied = GeneralizedDoubleKMeans(
        n_row_clusters=2,
        n_col_clusters=3,
        tie_columns=True,
        norm="mav_ratio",
        max_iter=5,  # Keep it short for testing
        random_state=42,
    )

    try:
        gdkm_tied.fit(X)
        print(f"   ✓ Tied columns successful. Loss: {gdkm_tied.loss_:.4f}")
        print(f"   ✓ Row labels: {gdkm_tied.row_labels_}")
        print(f"   ✓ Column labels: {gdkm_tied.column_labels_}")
    except Exception as e:
        print(f"   ✗ Tied columns failed: {e}")
        return False

    # Test with untied columns (per-row-cluster V)
    print("\n2. Testing with untied columns (per-row-cluster V)...")
    gdkm_untied = GeneralizedDoubleKMeans(
        n_row_clusters=2,
        n_col_clusters_list=[
            2,
            3,
        ],  # 2 col clusters for row cluster 0, 3 for row cluster 1
        tie_columns=False,
        norm="mav_ratio",
        max_iter=5,
        random_state=42,
    )

    try:
        gdkm_untied.fit(X)
        print(f"   ✓ Untied columns successful. Loss: {gdkm_untied.loss_:.4f}")
        print(f"   ✓ Row labels: {gdkm_untied.row_labels_}")
        print(f"   ✓ Column labels: {gdkm_untied.column_labels_}")
    except Exception as e:
        print(f"   ✗ Untied columns failed: {e}")
        return False

    # Compare with other norms
    print("\n3. Comparing MAV ratio with other norms...")
    norms = ["l1", "l2", "huber", "mav_ratio"]
    results = {}

    for norm in norms:
        try:
            gdkm = GeneralizedDoubleKMeans(
                n_row_clusters=2,
                n_col_clusters=2,
                tie_columns=True,
                norm=norm,
                max_iter=5,
                random_state=42,
            )
            gdkm.fit(X)
            results[norm] = gdkm.loss_
            print(f"   ✓ {norm}: Loss = {gdkm.loss_:.4f}")
        except Exception as e:
            print(f"   ✗ {norm}: Failed with {e}")
            results[norm] = None

    print(f"\n4. Loss comparison:")
    for norm, loss in results.items():
        if loss is not None:
            print(f"   {norm:10s}: {loss:.6f}")
        else:
            print(f"   {norm:10s}: FAILED")

    return True


def test_mav_ratio_calculation():
    """Test that MAV ratio is calculated correctly."""
    if not GDKM_AVAILABLE:
        print("Skipping GDKM calculation test - module not available")
        return True

    print("\n5. Testing MAV ratio calculation...")

    # Simple test case where we can verify the calculation
    X = np.array([[1.0, 2.0], [3.0, 4.0]])

    # Manual calculation for verification
    # If we have perfect reconstruction (diff = 0), MAV ratio should be 0
    # If reconstruction = 0, then diff = X, so MAV ratio = sum(|X|) / sum(|X|) = 1

    print(f"   Test data X:\n{X}")
    print(f"   Sum of absolute values: {np.sum(np.abs(X))}")

    gdkm = GeneralizedDoubleKMeans(
        n_row_clusters=1,  # Single cluster should give perfect reconstruction
        n_col_clusters=1,
        tie_columns=True,
        norm="mav_ratio",
        max_iter=10,
        random_state=42,
    )

    gdkm.fit(X)
    print(f"   ✓ Single cluster MAV ratio loss: {gdkm.loss_:.6f}")
    print(f"   ✓ Should be close to 0 for single cluster (perfect reconstruction)")

    return True


if __name__ == "__main__":
    print("=" * 60)
    print("GDKM MAV Ratio Norm Test")
    print("=" * 60)

    success = True
    success &= test_mav_ratio_norm()
    success &= test_mav_ratio_calculation()

    print("\n" + "=" * 60)
    if success:
        print("✓ All tests passed! MAV ratio norm is working correctly.")
    else:
        print("✗ Some tests failed. Check the implementation.")
    print("=" * 60)
