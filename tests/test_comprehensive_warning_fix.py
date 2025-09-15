#!/usr/bin/env python3
"""
Test comprehensive warning suppression for BTF grid search.

This script tests that all numpy warnings are properly suppressed.
"""

import numpy as np
import warnings
import sys
import os
import io
import contextlib

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

try:
    from src.BinaryTriFactorizationEstimator import BinaryTriFactorizationEstimator
    from src.BTNMF_util import sweep_btf_grid
except ImportError as e:
    print(f"Import error: {e}")
    sys.exit(1)


def create_sparse_problematic_data():
    """Create very sparse data that will trigger warnings."""
    np.random.seed(42)
    
    # Create extremely sparse binary data
    n_rows, n_cols = 40, 25
    X = np.random.rand(n_rows, n_cols) > 0.95  # Only 5% density
    X = X.astype(float)
    
    print(f"📊 Created very sparse data: {X.shape} with {X.sum():.0f} ones ({X.mean()*100:.1f}% density)")
    return X


def capture_all_output():
    """Context manager to capture both stdout and stderr."""
    stdout_capture = io.StringIO()
    stderr_capture = io.StringIO()
    
    @contextlib.contextmanager
    def capture():
        old_stdout = sys.stdout
        old_stderr = sys.stderr
        try:
            sys.stdout = stdout_capture
            sys.stderr = stderr_capture
            yield stdout_capture, stderr_capture
        finally:
            sys.stdout = old_stdout
            sys.stderr = old_stderr
    
    return capture()


def test_comprehensive_warning_suppression():
    """Test that all warnings are suppressed during BTF operations."""
    
    print("🔇 COMPREHENSIVE WARNING SUPPRESSION TEST")
    print("="*60)
    
    # Create problematic data
    X = create_sparse_problematic_data()
    
    # Create estimator maker
    est_maker = BinaryTriFactorizationEstimator.factory(
        loss="gaussian",
        max_iter=15,  # Reduced for faster testing
        tol=1e-3
    )
    
    print("\n🧪 Running BTF grid search with warning capture...")
    
    # Capture all warnings and output
    with warnings.catch_warnings(record=True) as warning_list:
        warnings.simplefilter("always")  # Capture all warnings
        
        # Also capture stdout/stderr to check for printed warnings
        with capture_all_output() as (stdout_capture, stderr_capture):
            try:
                results = sweep_btf_grid(
                    est_maker=est_maker,
                    X=X,
                    R_list=[5, 8, 11],    # Large R values likely to cause issues
                    C_list=[4, 7, 10],    # Large C values likely to cause issues  
                    restarts=1,           # Minimal for testing
                    min_keep=2,           # Low threshold
                    n_jobs=1,             # Single-threaded for cleaner capture
                )
                
                print(f"✅ Grid search completed: {len(results)} results")
                success = True
                
            except Exception as e:
                print(f"❌ Grid search failed: {e}")
                success = False
    
    # Get captured output
    stdout_content = stdout_capture.getvalue()
    stderr_content = stderr_capture.getvalue()
    
    # Analyze results
    print(f"\n📊 COMPREHENSIVE ANALYSIS:")
    print(f"   Total warnings captured: {len(warning_list)}")
    print(f"   Stdout lines: {len(stdout_content.splitlines())}")
    print(f"   Stderr lines: {len(stderr_content.splitlines())}")
    
    # Check for specific problematic warnings
    numpy_warnings = [w for w in warning_list if 'numpy' in str(w.filename).lower()]
    mean_warnings = [w for w in warning_list if 'mean of empty slice' in str(w.message).lower()]
    divide_warnings = [w for w in warning_list if 'invalid value encountered in divide' in str(w.message).lower()]
    runtime_warnings = [w for w in warning_list if w.category == RuntimeWarning]
    
    print(f"\n🔍 WARNING BREAKDOWN:")
    print(f"   NumPy warnings: {len(numpy_warnings)}")
    print(f"   'Mean of empty slice': {len(mean_warnings)}")
    print(f"   'Invalid value in divide': {len(divide_warnings)}")
    print(f"   Runtime warnings: {len(runtime_warnings)}")
    
    # Check stdout/stderr for warning text
    warning_keywords = ['RuntimeWarning', 'Mean of empty slice', 'invalid value encountered']
    stdout_warnings = sum(1 for keyword in warning_keywords if keyword in stdout_content)
    stderr_warnings = sum(1 for keyword in warning_keywords if keyword in stderr_content)
    
    print(f"\n📝 OUTPUT ANALYSIS:")
    print(f"   Warning keywords in stdout: {stdout_warnings}")
    print(f"   Warning keywords in stderr: {stderr_warnings}")
    
    # Show sample of any remaining warnings
    if warning_list:
        print(f"\n⚠️  REMAINING WARNINGS (first 3):")
        for i, w in enumerate(warning_list[:3]):
            print(f"   {i+1}. {w.category.__name__}: {w.message}")
            print(f"      File: {w.filename}:{w.lineno}")
    
    # Show sample of output if it contains warnings
    if stdout_warnings > 0 or stderr_warnings > 0:
        print(f"\n📄 SAMPLE OUTPUT WITH WARNINGS:")
        if stdout_content:
            lines = stdout_content.splitlines()
            warning_lines = [line for line in lines if any(kw in line for kw in warning_keywords)]
            for line in warning_lines[:3]:
                print(f"   STDOUT: {line}")
        if stderr_content:
            lines = stderr_content.splitlines()
            warning_lines = [line for line in lines if any(kw in line for kw in warning_keywords)]
            for line in warning_lines[:3]:
                print(f"   STDERR: {line}")
    
    # Determine success
    total_problematic = len(mean_warnings) + len(divide_warnings) + stdout_warnings + stderr_warnings
    
    if total_problematic == 0:
        print(f"\n🎉 PERFECT! No problematic warnings detected anywhere!")
        print(f"✅ Warning suppression is working comprehensively")
        return True
    else:
        print(f"\n⚠️  {total_problematic} problematic warnings still detected")
        print(f"🔧 Warning suppression needs more work")
        return False


def test_before_after_comparison():
    """Compare warning output before and after suppression."""
    
    print(f"\n🔄 BEFORE/AFTER COMPARISON")
    print(f"="*50)
    
    # Create simple test data
    X = np.random.rand(20, 15) > 0.9
    X = X.astype(float)
    
    est_maker = BinaryTriFactorizationEstimator.factory(
        loss="gaussian",
        max_iter=10,
        tol=1e-2
    )
    
    print("1️⃣ Testing with manual warning trigger (should show warnings)...")
    
    # Manually trigger the types of warnings we expect
    with warnings.catch_warnings(record=True) as manual_warnings:
        warnings.simplefilter("always")
        
        # Trigger empty slice warning
        empty_arr = np.array([])
        try:
            _ = np.mean(empty_arr)
        except:
            pass
        
        # Trigger divide warning  
        try:
            _ = np.array([1.0]) / np.array([0.0])
        except:
            pass
    
    print(f"   Manual triggers produced: {len(manual_warnings)} warnings")
    
    print("\n2️⃣ Testing BTF with small grid (should be clean)...")
    
    with warnings.catch_warnings(record=True) as btf_warnings:
        warnings.simplefilter("always")
        
        try:
            results = sweep_btf_grid(
                est_maker=est_maker,
                X=X,
                R_list=[2, 3],
                C_list=[2, 3],
                restarts=1,
                min_keep=1,
                n_jobs=1
            )
            print(f"   BTF grid search produced: {len(btf_warnings)} warnings")
            print(f"   Results: {len(results)} combinations processed")
            
        except Exception as e:
            print(f"   BTF failed: {e}")
            return False
    
    # Success if BTF produces fewer warnings than manual triggers
    if len(btf_warnings) <= len(manual_warnings):
        print(f"✅ BTF warning suppression is working!")
        return True
    else:
        print(f"❌ BTF still producing too many warnings")
        return False


def main():
    """Main test function."""
    
    print("🔇 COMPREHENSIVE BTF WARNING SUPPRESSION TEST")
    print("="*70)
    
    # Test 1: Before/after comparison
    comparison_passed = test_before_after_comparison()
    
    # Test 2: Comprehensive suppression
    comprehensive_passed = test_comprehensive_warning_suppression()
    
    # Summary
    print(f"\n🎯 FINAL SUMMARY")
    print(f"="*50)
    print(f"✅ Before/after comparison: {'PASS' if comparison_passed else 'FAIL'}")
    print(f"✅ Comprehensive suppression: {'PASS' if comprehensive_passed else 'FAIL'}")
    
    if comparison_passed and comprehensive_passed:
        print(f"\n🎉 ALL TESTS PASSED!")
        print(f"✅ BTF grid search warnings are comprehensively suppressed")
        print(f"🚀 Your multiprocessing BTF runs should now be clean and quiet")
    else:
        print(f"\n❌ SOME TESTS FAILED")
        if not comparison_passed:
            print(f"🔧 Basic warning suppression needs work")
        if not comprehensive_passed:
            print(f"🔧 Comprehensive suppression needs improvement")
    
    return comparison_passed and comprehensive_passed


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
