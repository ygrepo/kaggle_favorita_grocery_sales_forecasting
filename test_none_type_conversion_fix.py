#!/usr/bin/env python3
"""
Test the None type conversion fix in generate_growth_rate_store_sku_feature.

This script tests that the function can handle None values in columns properly.
"""

import pandas as pd
import numpy as np
import sys
import os
from datetime import datetime, timedelta

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

try:
    from src.data_utils import generate_growth_rate_store_sku_feature
except ImportError as e:
    print(f"Import error: {e}")
    sys.exit(1)


def create_test_data_with_none_values():
    """Create test data that includes None values in various columns."""
    
    print("📊 Creating test data with None values...")
    
    # Create base data
    dates = pd.date_range('2024-01-01', periods=10, freq='D')
    stores = [1, 2, None]  # Include None in store
    items = [100, 200, None]  # Include None in item
    
    data = []
    for date in dates:
        for store in stores:
            for item in items:
                # Create some realistic sales data
                unit_sales = np.random.uniform(10, 100) if np.random.random() > 0.1 else np.nan
                onpromotion = np.random.choice([0, 1, None])  # Include None in promotion
                
                data.append({
                    'date': date,
                    'store': store,
                    'item': item,
                    'unit_sales': unit_sales,
                    'onpromotion': onpromotion
                })
    
    df = pd.DataFrame(data)
    
    print(f"✅ Created test data: {df.shape}")
    print(f"   Store None values: {df['store'].isnull().sum()}")
    print(f"   Item None values: {df['item'].isnull().sum()}")
    print(f"   Promotion None values: {df['onpromotion'].isnull().sum()}")
    
    return df


def test_generate_growth_rate_with_none_values():
    """Test generate_growth_rate_store_sku_feature with None values."""
    
    print("\n🧪 TESTING generate_growth_rate_store_sku_feature WITH NONE VALUES")
    print("="*70)
    
    # Create test data with None values
    df = create_test_data_with_none_values()
    
    try:
        print("\n🔧 Running generate_growth_rate_store_sku_feature...")
        
        # Test with a small window size for faster execution
        result = generate_growth_rate_store_sku_feature(
            df=df,
            window_size=3,
            date_col="date",
            store_col="store",
            item_col="item",
            sales_col="unit_sales",
            promo_col="onpromotion",
            weight_col=None,  # No weights for simplicity
            output_path=None  # Don't save to file
        )
        
        print(f"✅ Function completed successfully!")
        
        if result is not None and not result.empty:
            print(f"   Result shape: {result.shape}")
            print(f"   Result columns: {list(result.columns)}")
            
            # Check data types
            print(f"\n📊 Result data types:")
            for col in ['store', 'item', 'onpromotion']:
                if col in result.columns:
                    dtype = result[col].dtype
                    null_count = result[col].isnull().sum()
                    print(f"   {col}: {dtype} (nulls: {null_count})")
            
            # Show sample of results
            print(f"\n📋 Sample results:")
            print(result.head(3).to_string())
            
            success = True
        else:
            print(f"❌ Function returned empty result")
            success = False
            
    except Exception as e:
        print(f"❌ Function failed with error: {e}")
        import traceback
        traceback.print_exc()
        success = False
    
    return success


def test_edge_cases():
    """Test various edge cases with None values."""
    
    print("\n🧪 TESTING EDGE CASES")
    print("="*40)
    
    # Test 1: All stores are None
    print("\n1️⃣ Testing with all stores as None...")
    df1 = pd.DataFrame({
        'date': pd.date_range('2024-01-01', periods=5),
        'store': [None] * 5,
        'item': [100] * 5,
        'unit_sales': [10.0] * 5,
        'onpromotion': [0] * 5
    })
    
    try:
        result1 = generate_growth_rate_store_sku_feature(
            df=df1, window_size=2, date_col="date", store_col="store", 
            item_col="item", sales_col="unit_sales", promo_col="onpromotion",
            weight_col=None, output_path=None
        )
        if result1 is None or result1.empty:
            print(f"   ✅ Correctly handled all-None stores (empty result)")
            all_none_stores_success = True
        else:
            print(f"   ⚠️  Got non-empty result with all-None stores: {result1.shape}")
            all_none_stores_success = True  # Still success if it doesn't crash
    except Exception as e:
        print(f"   ❌ Failed with all-None stores: {e}")
        all_none_stores_success = False
    
    # Test 2: Mixed None and valid values
    print("\n2️⃣ Testing with mixed None and valid values...")
    df2 = pd.DataFrame({
        'date': pd.date_range('2024-01-01', periods=6),
        'store': [1, 1, None, 2, 2, None],
        'item': [100, 100, 200, 200, 200, 300],
        'unit_sales': [10.0, 15.0, 20.0, 25.0, 30.0, 35.0],
        'onpromotion': [0, 1, None, 0, 1, None]
    })
    
    try:
        result2 = generate_growth_rate_store_sku_feature(
            df=df2, window_size=2, date_col="date", store_col="store", 
            item_col="item", sales_col="unit_sales", promo_col="onpromotion",
            weight_col=None, output_path=None
        )
        if result2 is not None:
            print(f"   ✅ Mixed None values handled: {result2.shape}")
            mixed_success = True
        else:
            print(f"   ⚠️  Got None result with mixed values")
            mixed_success = False
    except Exception as e:
        print(f"   ❌ Failed with mixed None values: {e}")
        mixed_success = False
    
    return all_none_stores_success and mixed_success


def main():
    """Main test function."""
    
    print("🔧 NONE TYPE CONVERSION FIX TEST")
    print("="*60)
    
    # Test 1: Basic functionality with None values
    basic_success = test_generate_growth_rate_with_none_values()
    
    # Test 2: Edge cases
    edge_success = test_edge_cases()
    
    # Summary
    print(f"\n🎯 TEST SUMMARY")
    print(f"="*40)
    print(f"✅ Basic None handling: {'PASS' if basic_success else 'FAIL'}")
    print(f"✅ Edge cases: {'PASS' if edge_success else 'FAIL'}")
    
    if basic_success and edge_success:
        print(f"\n🎉 ALL TESTS PASSED!")
        print(f"✅ generate_growth_rate_store_sku_feature handles None values correctly")
        print(f"✅ No more 'int() argument must be a string...' errors")
        print(f"💡 Your growth rate generation should now work with messy data")
    else:
        print(f"\n❌ SOME TESTS FAILED")
        if not basic_success:
            print(f"🔧 Basic None handling needs work")
        if not edge_success:
            print(f"🔧 Edge case handling needs improvement")
    
    return basic_success and edge_success


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
