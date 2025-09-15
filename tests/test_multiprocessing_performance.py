#!/usr/bin/env python3
"""
Test multiprocessing performance for generate_growth_rate_features.

This script compares single-threaded vs multi-threaded performance.
"""

import pandas as pd
import numpy as np
import time
import sys
import os
from pathlib import Path

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

try:
    from src.data_utils import generate_growth_rate_features
except ImportError as e:
    print(f"Import error: {e}")
    sys.exit(1)


def create_test_data(n_stores=5, n_items=20, n_days=90):
    """Create test data for multiprocessing comparison."""
    print(f"📊 Creating test data: {n_stores} stores, {n_items} items, {n_days} days")
    
    dates = pd.date_range(start="2023-01-01", periods=n_days, freq='D')
    
    data = []
    for store in range(1, n_stores + 1):
        for item in range(1000, 1000 + n_items):
            for date in dates:
                # Simulate realistic sales patterns
                base_sales = np.random.poisson(8)
                seasonal_factor = 1 + 0.2 * np.sin(2 * np.pi * date.dayofyear / 365)
                weekend_boost = 1.3 if date.weekday() >= 5 else 1.0
                
                unit_sales = max(0, int(base_sales * seasonal_factor * weekend_boost))
                
                # Random promotions
                onpromotion = 1 if np.random.random() < 0.15 else 0
                if onpromotion:
                    unit_sales = int(unit_sales * 1.4)
                
                data.append({
                    'store': store,
                    'item': item,
                    'date': date,
                    'unit_sales': unit_sales,
                    'onpromotion': onpromotion,
                    'weight': np.random.uniform(0.9, 1.1)
                })
    
    df = pd.DataFrame(data)
    total_combinations = len(df[['store', 'item']].drop_duplicates())
    print(f"📈 Generated {len(df):,} rows with {total_combinations} store-item combinations")
    return df


def test_multiprocessing_performance():
    """Test performance comparison between single and multi-threaded processing."""
    
    print("🚀 MULTIPROCESSING PERFORMANCE TEST")
    print("="*60)
    
    # Create test data
    df = create_test_data(n_stores=4, n_items=15, n_days=60)
    
    # Test parameters
    window_size = 7
    
    print(f"\n🧪 Testing with window_size={window_size}")
    print(f"📊 Dataset: {len(df):,} rows")
    
    results = {}
    
    # Test 1: Single-threaded (n_jobs=1)
    print(f"\n⏱️  Single-threaded processing...")
    start_time = time.perf_counter()
    
    result_single = generate_growth_rate_features(
        df=df,
        window_size=window_size,
        calendar_aligned=True,
        n_jobs=1,  # Single-threaded
        log_level="WARNING"  # Reduce logging noise
    )
    
    single_time = time.perf_counter() - start_time
    results['single'] = {
        'time': single_time,
        'rows': len(result_single)
    }
    
    print(f"✅ Single-threaded: {single_time:.2f}s, {len(result_single)} rows")
    
    # Test 2: Multi-threaded (n_jobs=-1, all cores)
    print(f"\n🚀 Multi-threaded processing (all cores)...")
    start_time = time.perf_counter()
    
    result_multi = generate_growth_rate_features(
        df=df,
        window_size=window_size,
        calendar_aligned=True,
        n_jobs=-1,  # Use all CPU cores
        batch_size=25,  # Smaller batches for better load balancing
        log_level="WARNING"
    )
    
    multi_time = time.perf_counter() - start_time
    results['multi'] = {
        'time': multi_time,
        'rows': len(result_multi)
    }
    
    print(f"✅ Multi-threaded: {multi_time:.2f}s, {len(result_multi)} rows")
    
    # Test 3: Multi-threaded with specific core count
    n_cores = min(4, os.cpu_count())  # Use up to 4 cores
    print(f"\n🔧 Multi-threaded processing ({n_cores} cores)...")
    start_time = time.perf_counter()
    
    result_multi_4 = generate_growth_rate_features(
        df=df,
        window_size=window_size,
        calendar_aligned=True,
        n_jobs=n_cores,
        batch_size=20,
        log_level="WARNING"
    )
    
    multi_4_time = time.perf_counter() - start_time
    results['multi_4'] = {
        'time': multi_4_time,
        'rows': len(result_multi_4)
    }
    
    print(f"✅ Multi-threaded ({n_cores} cores): {multi_4_time:.2f}s, {len(result_multi_4)} rows")
    
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
        print("✅ All versions produced the same number of rows")
    else:
        print("⚠️  Row count mismatch - check implementation")
    
    # Recommendations
    print(f"\n💡 RECOMMENDATIONS:")
    print(f"{'='*40}")
    
    if speedup_all > 2.0:
        print("🚀 Multiprocessing provides significant speedup!")
        print(f"   Recommended: n_jobs=-1 (use all {cpu_count} cores)")
    elif speedup_4 > 1.5:
        print("⚡ Moderate speedup with multiprocessing")
        print(f"   Recommended: n_jobs={n_cores} for balanced performance")
    else:
        print("🤔 Limited speedup - consider:")
        print("   - Larger datasets (more store-item combinations)")
        print("   - Adjusting batch_size parameter")
        print("   - Single-threaded may be sufficient for small datasets")
    
    return results


def test_batch_size_optimization():
    """Test different batch sizes to find optimal configuration."""
    
    print(f"\n🔧 BATCH SIZE OPTIMIZATION")
    print(f"{'='*50}")
    
    # Create test data
    df = create_test_data(n_stores=3, n_items=20, n_days=45)
    
    batch_sizes = [10, 25, 50, 100]
    n_cores = min(4, os.cpu_count())
    
    print(f"Testing batch sizes: {batch_sizes}")
    print(f"Using {n_cores} cores")
    
    batch_results = {}
    
    for batch_size in batch_sizes:
        print(f"\n🧪 Testing batch_size={batch_size}...")
        
        start_time = time.perf_counter()
        result = generate_growth_rate_features(
            df=df,
            window_size=7,
            calendar_aligned=True,
            n_jobs=n_cores,
            batch_size=batch_size,
            log_level="ERROR"  # Minimal logging
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
    results = test_multiprocessing_performance()
    
    # Test batch size optimization
    test_batch_size_optimization()
    
    print(f"\n🎯 SUMMARY:")
    print(f"{'='*40}")
    print("✅ Multiprocessing implementation complete")
    print("✅ Performance testing complete")
    print("💡 Use n_jobs=-1 for maximum performance on multi-core systems")
    print("💡 Adjust batch_size based on your dataset size and available memory")


if __name__ == "__main__":
    main()
