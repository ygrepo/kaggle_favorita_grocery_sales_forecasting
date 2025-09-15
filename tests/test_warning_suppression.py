#!/usr/bin/env python3
"""
Test that numpy warnings are properly suppressed in BTF grid search.

This script tests the warning suppression for empty slice operations.
"""

import numpy as np
import warnings
import sys
import os

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

try:
    from src.BinaryTriFactorizationEstimator import BinaryTriFactorizationEstimator
    from src.BTNMF_util import sweep_btf_grid
except ImportError as e:
    print(f"Import error: {e}")
    sys.exit(1)


def create_problematic_data():
    """Create data that is likely to trigger empty slice warnings."""
    np.random.seed(42)
    
    # Create very sparse binary data that might lead to empty blocks
    n_rows, n_cols = 30, 20
    X = np.random.rand(n_rows, n_cols) > 0.9  # Very sparse (10% density)
    X = X.astype(float)
    
    print(f"📊 Created sparse data: {X.shape} with {X.sum():.0f} ones ({X.mean()*100:.1f}% density)")
    return X


def test_warning_suppression():
    """Test that warnings are properly suppressed during BTF grid search."""
    
    print("🔇 TESTING WARNING SUPPRESSION")
    print("="*50)
    
    # Create sparse data that might trigger warnings
    X = create_problematic_data()
    
    # Create estimator maker
    est_maker = BinaryTriFactorizationEstimator.factory(
        loss="gaussian",
        max_iter=20,  # Reduced for faster testing
        tol=1e-3
    )
    
    # Capture warnings
    with warnings.catch_warnings(record=True) as warning_list:
        warnings.simplefilter("always")  # Capture all warnings
        
        print("\n🧪 Running BTF grid search (should suppress numpy warnings)...")
        
        try:
            results = sweep_btf_grid(
                est_maker=est_maker,
                X=X,
                R_list=[3, 5, 8],  # Larger R values more likely to cause issues
                C_list=[2, 4, 6],  # Larger C values more likely to cause issues
                restarts=1,  # Minimal for testing
                min_keep=3,  # Lower threshold
                n_jobs=1,    # Single-threaded for cleaner warning capture
            )
            
            print(f"✅ Grid search completed: {len(results)} results")
            
        except Exception as e:
            print(f"❌ Grid search failed: {e}")
            return False
    
    # Analyze captured warnings
    numpy_warnings = [w for w in warning_list if 'numpy' in str(w.filename).lower()]
    mean_warnings = [w for w in warning_list if 'mean of empty slice' in str(w.message).lower()]
    divide_warnings = [w for w in warning_list if 'invalid value encountered in divide' in str(w.message).lower()]
    
    print(f"\n📊 WARNING ANALYSIS:")
    print(f"   Total warnings captured: {len(warning_list)}")
    print(f"   NumPy warnings: {len(numpy_warnings)}")
    print(f"   'Mean of empty slice' warnings: {len(mean_warnings)}")
    print(f"   'Invalid value in divide' warnings: {len(divide_warnings)}")
    
    # Show sample warnings if any
    if warning_list:
        print(f"\n📝 Sample warnings:")
        for i, w in enumerate(warning_list[:3]):  # Show first 3 warnings
            print(f"   {i+1}. {w.category.__name__}: {w.message}")
            print(f"      File: {w.filename}:{w.lineno}")
    
    # Check if problematic warnings were suppressed
    problematic_warnings = mean_warnings + divide_warnings
    
    if len(problematic_warnings) == 0:
        print(f"\n✅ SUCCESS: No problematic numpy warnings detected!")
        print(f"   The warning suppression is working correctly.")
        return True
    else:
        print(f"\n⚠️  WARNING: {len(problematic_warnings)} problematic warnings still present:")
        for w in problematic_warnings:
            print(f"   - {w.message}")
        print(f"   Consider adding more warning suppression.")
        return False


def test_manual_warning_trigger():
    """Test that we can manually trigger and suppress the warnings."""
    
    print(f"\n🧪 MANUAL WARNING TRIGGER TEST")
    print(f"="*50)
    
    print("1️⃣ Testing without suppression (should show warnings)...")
    
    with warnings.catch_warnings(record=True) as w1:
        warnings.simplefilter("always")
        
        # This should trigger warnings
        empty_array = np.array([])
        try:
            result = np.mean(empty_array)  # Should warn about empty slice
            print(f"   Result: {result}")
        except:
            pass
        
        try:
            result = np.array([1.0]) / np.array([0.0])  # Should warn about divide
            print(f"   Result: {result}")
        except:
            pass
    
    print(f"   Warnings without suppression: {len(w1)}")
    
    print("\n2️⃣ Testing with suppression (should be silent)...")
    
    with warnings.catch_warnings(record=True) as w2:
        warnings.simplefilter("always")
        
        # This should NOT trigger warnings due to suppression
        with np.errstate(invalid='ignore', divide='ignore'):
            empty_array = np.array([])
            try:
                result = np.mean(empty_array)  # Should be silent
                print(f"   Result: {result}")
            except:
                pass
            
            try:
                result = np.array([1.0]) / np.array([0.0])  # Should be silent
                print(f"   Result: {result}")
            except:
                pass
    
    print(f"   Warnings with suppression: {len(w2)}")
    
    if len(w1) > len(w2):
        print(f"✅ Warning suppression is working: {len(w1)} -> {len(w2)} warnings")
        return True
    else:
        print(f"❌ Warning suppression may not be working properly")
        return False


def main():
    """Main test function."""
    
    print("🔇 NUMPY WARNING SUPPRESSION TEST")
    print("="*60)
    
    # Test manual warning suppression
    manual_test_passed = test_manual_warning_trigger()
    
    # Test BTF warning suppression
    btf_test_passed = test_warning_suppression()
    
    # Summary
    print(f"\n🎯 TEST SUMMARY")
    print(f"="*40)
    print(f"✅ Manual suppression test: {'PASS' if manual_test_passed else 'FAIL'}")
    print(f"✅ BTF suppression test: {'PASS' if btf_test_passed else 'FAIL'}")
    
    if manual_test_passed and btf_test_passed:
        print(f"\n🎉 ALL TESTS PASSED!")
        print(f"✅ NumPy warnings are properly suppressed in BTF operations")
        print(f"💡 Your BTF grid searches should now run without annoying warnings")
    else:
        print(f"\n❌ SOME TESTS FAILED")
        print(f"🔧 Warning suppression may need additional work")
    
    return manual_test_passed and btf_test_passed


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
