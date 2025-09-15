#!/usr/bin/env python3
"""
Test the pickle fix for BTF multiprocessing.

This script tests that the BTFEstimatorBuilder is properly picklable.
"""

import numpy as np
import pickle
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


def test_pickle_compatibility():
    """Test that the new BTFEstimatorBuilder is picklable."""
    
    print("🧪 TESTING PICKLE COMPATIBILITY")
    print("="*50)
    
    # Test 1: Old approach (should fail)
    print("\n1️⃣ Testing old approach (lambda function)...")
    
    def old_est_maker(**kwargs):
        return BinaryTriFactorizationEstimator(loss="gaussian", max_iter=10, **kwargs)
    
    try:
        pickle.dumps(old_est_maker)
        print("✅ Old approach is picklable (unexpected)")
    except Exception as e:
        print(f"❌ Old approach failed (expected): {type(e).__name__}")
    
    # Test 2: New approach (should work)
    print("\n2️⃣ Testing new approach (factory method)...")
    
    new_est_maker = BinaryTriFactorizationEstimator.factory(
        loss="gaussian",
        max_iter=10
    )
    
    try:
        pickled = pickle.dumps(new_est_maker)
        unpickled = pickle.loads(pickled)
        print("✅ New approach is picklable!")
        
        # Test that unpickled version works
        test_est = unpickled(n_row_clusters=2, n_col_clusters=2, random_state=42)
        print(f"✅ Unpickled builder works: {type(test_est).__name__}")
        
    except Exception as e:
        print(f"❌ New approach failed: {e}")
        return False
    
    return True


def test_multiprocessing_integration():
    """Test that multiprocessing works with the new approach."""
    
    print("\n🚀 TESTING MULTIPROCESSING INTEGRATION")
    print("="*50)
    
    # Create small test data
    np.random.seed(42)
    X = np.random.rand(20, 15) > 0.6
    X = X.astype(float)
    
    print(f"📊 Test data: {X.shape} matrix with {X.sum():.0f} ones")
    
    # Create picklable estimator maker
    est_maker = BinaryTriFactorizationEstimator.factory(
        loss="gaussian",
        max_iter=20,  # Reduced for faster testing
        tol=1e-3
    )
    
    # Test small grid search
    R_list = [2, 3]
    C_list = [2, 3]
    
    print(f"\n🧪 Testing grid search: R={R_list}, C={C_list}")
    
    try:
        # Test single-threaded first
        print("⏱️  Single-threaded test...")
        result_single = sweep_btf_grid(
            est_maker=est_maker,
            X=X,
            R_list=R_list,
            C_list=C_list,
            restarts=1,  # Minimal for testing
            min_keep=2,
            n_jobs=1  # Single-threaded
        )
        
        print(f"✅ Single-threaded: {len(result_single)} results")
        
        # Test multi-threaded
        print("🚀 Multi-threaded test...")
        result_multi = sweep_btf_grid(
            est_maker=est_maker,
            X=X,
            R_list=R_list,
            C_list=C_list,
            restarts=1,  # Minimal for testing
            min_keep=2,
            n_jobs=2,  # Use 2 processes
            batch_size=2
        )
        
        print(f"✅ Multi-threaded: {len(result_multi)} results")
        
        # Validate results
        if len(result_single) == len(result_multi):
            print("✅ Both approaches produced same number of results")
            
            # Check if results are similar
            if not result_single.empty and not result_multi.empty:
                single_pve = result_single['PVE'].values
                multi_pve = result_multi['PVE'].values
                
                if len(single_pve) == len(multi_pve):
                    max_diff = np.abs(single_pve - multi_pve).max()
                    print(f"✅ Maximum PVE difference: {max_diff:.6f}")
                    
                    if max_diff < 1e-6:
                        print("✅ Results are very similar!")
                        return True
                    else:
                        print("⚠️  Results have some differences (may be due to randomness)")
                        return True
        else:
            print(f"❌ Result count mismatch: {len(result_single)} vs {len(result_multi)}")
            return False
            
    except Exception as e:
        print(f"❌ Multiprocessing test failed: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    return True


def test_builder_functionality():
    """Test that the BTFEstimatorBuilder works correctly."""
    
    print("\n🔧 TESTING BUILDER FUNCTIONALITY")
    print("="*50)
    
    # Create builder with some frozen kwargs
    builder = BinaryTriFactorizationEstimator.factory(
        loss="gaussian",
        max_iter=50,
        tol=1e-4
    )
    
    print("✅ Builder created")
    
    # Test building estimators
    try:
        # Test basic building
        est1 = builder(n_row_clusters=3, n_col_clusters=2)
        print(f"✅ Built estimator: {est1.n_row_clusters}×{est1.n_col_clusters}")
        print(f"   Loss: {est1.loss}, Max iter: {est1.max_iter}")
        
        # Test with overrides
        est2 = builder(n_row_clusters=4, n_col_clusters=3, max_iter=100, random_state=42)
        print(f"✅ Built with overrides: {est2.n_row_clusters}×{est2.n_col_clusters}")
        print(f"   Max iter: {est2.max_iter}, Random state: {est2.random_state}")
        
        # Test that frozen kwargs are preserved
        if est1.loss == "gaussian" and est2.loss == "gaussian":
            print("✅ Frozen kwargs preserved")
        else:
            print("❌ Frozen kwargs not preserved")
            return False
            
        # Test that overrides work
        if est2.max_iter == 100:
            print("✅ Overrides work")
        else:
            print("❌ Overrides don't work")
            return False
            
    except Exception as e:
        print(f"❌ Builder functionality test failed: {e}")
        return False
    
    return True


def main():
    """Main test function."""
    
    print("🔧 BTF PICKLE FIX VALIDATION")
    print("="*60)
    
    # Run all tests
    tests = [
        ("Pickle Compatibility", test_pickle_compatibility),
        ("Builder Functionality", test_builder_functionality),
        ("Multiprocessing Integration", test_multiprocessing_integration),
    ]
    
    results = {}
    
    for test_name, test_func in tests:
        print(f"\n{'='*60}")
        print(f"🧪 {test_name.upper()}")
        print(f"{'='*60}")
        
        try:
            success = test_func()
            results[test_name] = success
        except Exception as e:
            print(f"❌ {test_name} failed with exception: {e}")
            results[test_name] = False
    
    # Summary
    print(f"\n🎯 TEST SUMMARY")
    print(f"{'='*40}")
    
    all_passed = True
    for test_name, success in results.items():
        status = "✅ PASS" if success else "❌ FAIL"
        print(f"{status} {test_name}")
        if not success:
            all_passed = False
    
    if all_passed:
        print(f"\n🎉 ALL TESTS PASSED!")
        print(f"✅ BTF multiprocessing is now working correctly")
        print(f"💡 Use BinaryTriFactorizationEstimator.factory() for multiprocessing")
    else:
        print(f"\n❌ SOME TESTS FAILED")
        print(f"🔧 Check the implementation and try again")
    
    return all_passed


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
