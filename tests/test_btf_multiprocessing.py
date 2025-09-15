#!/usr/bin/env python3
"""
Test multiprocessing performance for sweep_btf_grid.

This script compares single-threaded vs multi-threaded performance for BTF grid search.
"""

import numpy as np
import pandas as pd
import time
import sys
import os
from pathlib import Path

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

try:
    from src.BinaryTriFactorizationEstimator import BinaryTriFactorizationEstimator
    from src.BTNMF_util import sweep_btf_grid
except ImportError as e:
    print(f"Import error: {e}")
    sys.exit(1)


def create_test_data(n_rows=50, n_cols=30, noise_level=0.1):
    """Create synthetic binary data for BTF testing."""
    print(f"📊 Creating test data: {n_rows}×{n_cols} matrix")
    
    # Create structured binary data with some block structure
    np.random.seed(42)  # For reproducibility
    
    # Create block structure
    true_R, true_C = 3, 2
    U = np.random.rand(n_rows, true_R) > 0.7  # Sparse row assignments
    V = np.random.rand(n_cols, true_C) > 0.6  # Sparse col assignments
    B = np.random.rand(true_R, true_C) > 0.5  # Block values
    
    # Generate structured data
    X_structured = (U @ B @ V.T).astype(float)
    
    # Add noise
    noise = np.random.rand(n_rows, n_cols) < noise_level
    X = (X_structured + noise) > 0.5
    X = X.astype(float)
    
    print(f"📈 Generated {n_rows}×{n_cols} binary matrix with {X.sum():.0f} ones ({X.mean()*100:.1f}% density)")
    return X


def test_btf_multiprocessing_performance():
    """Test performance comparison between single and multi-threaded BTF grid search."""
    
    print("🚀 BTF MULTIPROCESSING PERFORMANCE TEST")
    print("="*60)
    
    # Create test data
    X = create_test_data(n_rows=40, n_cols=25)
    
    # Define grid search parameters
    R_list = range(2, 5)  # [2, 3, 4]
    C_list = range(2, 4)  # [2, 3]
    total_combinations = len(R_list) * len(C_list)
    
    print(f"\n🧪 Grid search parameters:")
    print(f"   R_list: {list(R_list)}")
    print(f"   C_list: {list(C_list)}")
    print(f"   Total combinations: {total_combinations}")
    
    # Estimator maker function
    def est_maker(**kwargs):
        return BinaryTriFactorizationEstimator(
            loss="gaussian",
            max_iter=50,  # Reduced for faster testing
            **kwargs
        )
    
    results = {}
    
    # Test 1: Single-threaded (n_jobs=1)
    print(f"\n⏱️  Single-threaded processing...")
    start_time = time.perf_counter()
    
    result_single = sweep_btf_grid(
        est_maker=est_maker,
        X=X,
        R_list=R_list,
        C_list=C_list,
        restarts=2,  # Reduced for faster testing
        min_keep=3,  # Reduced for faster testing
        n_jobs=1,  # Single-threaded
    )
    
    single_time = time.perf_counter() - start_time
    results['single'] = {
        'time': single_time,
        'rows': len(result_single)
    }
    
    print(f"✅ Single-threaded: {single_time:.2f}s, {len(result_single)} result rows")
    
    # Test 2: Multi-threaded (n_jobs=-1, all cores)
    print(f"\n🚀 Multi-threaded processing (all cores)...")
    start_time = time.perf_counter()
    
    result_multi = sweep_btf_grid(
        est_maker=est_maker,
        X=X,
        R_list=R_list,
        C_list=C_list,
        restarts=2,  # Reduced for faster testing
        min_keep=3,  # Reduced for faster testing
        n_jobs=-1,  # Use all CPU cores
        batch_size=2,  # Small batches for better load balancing
    )
    
    multi_time = time.perf_counter() - start_time
    results['multi'] = {
        'time': multi_time,
        'rows': len(result_multi)
    }
    
    print(f"✅ Multi-threaded: {multi_time:.2f}s, {len(result_multi)} result rows")
    
    # Test 3: Multi-threaded with specific core count
    n_cores = min(4, os.cpu_count())  # Use up to 4 cores
    print(f"\n🔧 Multi-threaded processing ({n_cores} cores)...")
    start_time = time.perf_counter()
    
    result_multi_4 = sweep_btf_grid(
        est_maker=est_maker,
        X=X,
        R_list=R_list,
        C_list=C_list,
        restarts=2,
        min_keep=3,
        n_jobs=n_cores,
        batch_size=2,
    )
    
    multi_4_time = time.perf_counter() - start_time
    results['multi_4'] = {
        'time': multi_4_time,
        'rows': len(result_multi_4)
    }
    
    print(f"✅ Multi-threaded ({n_cores} cores): {multi_4_time:.2f}s, {len(result_multi_4)} result rows")
    
    # Performance Analysis
    print(f"\n📊 PERFORMANCE ANALYSIS")
    print(f"{'='*50}")
    
    single_time = results['single']['time']
    multi_time = results['multi']['time']
    multi_4_time = results['multi_4']['time']
    
    if single_time > 0:
        speedup_all = single_time / multi_time
        speedup_4 = single_time / multi_4_time
        
        print(f"⏱️  Execution Times:")
        print(f"   Single-threaded:     {single_time:.2f}s")
        print(f"   Multi-threaded (all): {multi_time:.2f}s")
        print(f"   Multi-threaded ({n_cores}):   {multi_4_time:.2f}s")
        
        print(f"\n🚀 Speedup:")
        print(f"   All cores:  {speedup_all:.1f}x faster")
        print(f"   {n_cores} cores:    {speedup_4:.1f}x faster")
        
        print(f"\n💡 Efficiency:")
        cpu_count = os.cpu_count()
        efficiency_all = speedup_all / cpu_count * 100
        efficiency_4 = speedup_4 / n_cores * 100
        print(f"   All cores ({cpu_count}):  {efficiency_all:.1f}% efficiency")
        print(f"   {n_cores} cores:    {efficiency_4:.1f}% efficiency")
    
    # Validation
    print(f"\n✅ VALIDATION:")
    print(f"   Single-threaded: {results['single']['rows']} rows")
    print(f"   Multi-threaded:  {results['multi']['rows']} rows")
    print(f"   4-core:          {results['multi_4']['rows']} rows")
    
    if (results['single']['rows'] == results['multi']['rows'] == results['multi_4']['rows']):
        print("✅ All versions produced the same number of result rows")
        
        # Check if results are similar (allowing for small numerical differences)
        if not result_single.empty and not result_multi.empty:
            # Compare key metrics
            single_pve = result_single['PVE'].values
            multi_pve = result_multi['PVE'].values
            
            if len(single_pve) == len(multi_pve):
                pve_diff = np.abs(single_pve - multi_pve).max()
                print(f"✅ Maximum PVE difference: {pve_diff:.6f} (should be ~0)")
                
                if pve_diff < 1e-10:
                    print("✅ Results are numerically identical")
                elif pve_diff < 1e-6:
                    print("✅ Results are very similar (small numerical differences)")
                else:
                    print("⚠️  Results have noticeable differences - check implementation")
    else:
        print("⚠️  Row count mismatch - check implementation")
    
    # Show sample results
    print(f"\n📊 SAMPLE RESULTS:")
    print("Single-threaded results:")
    print(result_single[['n_row', 'n_col', 'PVE', 'Mean Silhouette', 'AIC', 'BIC']].head())
    
    return results


def test_batch_size_optimization():
    """Test different batch sizes to find optimal configuration."""
    
    print(f"\n🔧 BATCH SIZE OPTIMIZATION")
    print(f"{'='*50}")
    
    # Create smaller test data for batch size testing
    X = create_test_data(n_rows=30, n_cols=20)
    
    # Smaller grid for batch size testing
    R_list = range(2, 4)  # [2, 3]
    C_list = range(2, 4)  # [2, 3]
    
    def est_maker(**kwargs):
        return BinaryTriFactorizationEstimator(loss="gaussian", max_iter=30, **kwargs)
    
    batch_sizes = [1, 2, 4]
    n_cores = min(4, os.cpu_count())
    
    print(f"Testing batch sizes: {batch_sizes}")
    print(f"Using {n_cores} cores")
    
    batch_results = {}
    
    for batch_size in batch_sizes:
        print(f"\n🧪 Testing batch_size={batch_size}...")
        
        start_time = time.perf_counter()
        result = sweep_btf_grid(
            est_maker=est_maker,
            X=X,
            R_list=R_list,
            C_list=C_list,
            restarts=2,
            min_keep=2,
            n_jobs=n_cores,
            batch_size=batch_size,
        )
        execution_time = time.perf_counter() - start_time
        
        batch_results[batch_size] = {
            'time': execution_time,
            'rows': len(result)
        }
        
        print(f"   {execution_time:.2f}s, {len(result)} rows")
    
    # Find optimal batch size
    best_batch_size = min(batch_results.keys(), key=lambda x: batch_results[x]['time'])
    best_time = batch_results[best_batch_size]['time']
    
    print(f"\n🏆 OPTIMAL BATCH SIZE: {best_batch_size}")
    print(f"   Best time: {best_time:.2f}s")
    
    print(f"\n📊 Batch Size Performance:")
    for batch_size, result in batch_results.items():
        relative_perf = result['time'] / best_time
        status = "🏆" if batch_size == best_batch_size else "  "
        print(f"   {status} {batch_size:3d}: {result['time']:.2f}s ({relative_perf:.1f}x)")


def main():
    """Main test function."""
    
    # Test multiprocessing performance
    results = test_btf_multiprocessing_performance()
    
    # Test batch size optimization
    test_batch_size_optimization()
    
    print(f"\n🎯 SUMMARY:")
    print(f"{'='*40}")
    print("✅ BTF multiprocessing implementation complete")
    print("✅ Performance testing complete")
    print("💡 Use n_jobs=-1 for maximum performance on multi-core systems")
    print("💡 Adjust batch_size based on your grid size and available cores")
    print("💡 Smaller batch_size = better load balancing, larger = less overhead")


if __name__ == "__main__":
    main()
