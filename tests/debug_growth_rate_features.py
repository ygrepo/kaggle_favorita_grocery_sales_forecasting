#!/usr/bin/env python3
"""
Debug script for generate_growth_rate_features "No growth rate features generated" issue.

This script helps identify why the function returns empty results.
"""

import pandas as pd
import numpy as np
import sys
import os
from pathlib import Path
import logging

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

try:
    from src.data_utils import generate_growth_rate_features, generate_growth_rate_store_sku_feature
except ImportError as e:
    print(f"Import error: {e}")
    sys.exit(1)


def debug_data_structure(df: pd.DataFrame):
    """Debug the input data structure."""
    print("🔍 DATA STRUCTURE ANALYSIS")
    print("="*50)
    
    print(f"📊 DataFrame shape: {df.shape}")
    print(f"📊 Columns: {list(df.columns)}")
    print(f"📊 Data types:\n{df.dtypes}")
    
    # Check required columns
    required_cols = ['store', 'item', 'date', 'unit_sales']
    missing_cols = [col for col in required_cols if col not in df.columns]
    if missing_cols:
        print(f"❌ Missing required columns: {missing_cols}")
        return False
    else:
        print(f"✅ All required columns present: {required_cols}")
    
    # Check for null values
    print(f"\n📊 Null values:")
    null_counts = df.isnull().sum()
    for col, count in null_counts.items():
        if count > 0:
            print(f"   {col}: {count} nulls ({count/len(df)*100:.1f}%)")
    
    # Check unique combinations
    store_item_combos = df[['store', 'item']].drop_duplicates()
    print(f"\n📊 Unique store-item combinations: {len(store_item_combos)}")
    print(f"📊 Unique stores: {df['store'].nunique()}")
    print(f"📊 Unique items: {df['item'].nunique()}")
    
    # Check date range
    if 'date' in df.columns:
        df['date'] = pd.to_datetime(df['date'])
        print(f"📊 Date range: {df['date'].min()} to {df['date'].max()}")
        print(f"📊 Total days: {(df['date'].max() - df['date'].min()).days + 1}")
    
    # Check sales data
    if 'unit_sales' in df.columns:
        print(f"📊 Unit sales stats:")
        print(f"   Min: {df['unit_sales'].min()}")
        print(f"   Max: {df['unit_sales'].max()}")
        print(f"   Mean: {df['unit_sales'].mean():.2f}")
        print(f"   Zero sales: {(df['unit_sales'] == 0).sum()} ({(df['unit_sales'] == 0).sum()/len(df)*100:.1f}%)")
    
    return True


def test_single_store_item(df: pd.DataFrame):
    """Test with a single store-item combination."""
    print("\n🧪 SINGLE STORE-ITEM TEST")
    print("="*50)
    
    # Get first store-item combination
    first_combo = df[['store', 'item']].drop_duplicates().iloc[0]
    store, item = first_combo['store'], first_combo['item']
    
    print(f"Testing store: {store}, item: {item}")
    
    # Extract data for this combination
    mask = (df['store'] == store) & (df['item'] == item)
    single_df = df[mask].copy()
    
    print(f"📊 Data for this combination: {len(single_df)} rows")
    if len(single_df) == 0:
        print("❌ No data found for this store-item combination")
        return None
    
    print(f"📊 Date range: {single_df['date'].min()} to {single_df['date'].max()}")
    print(f"📊 Sales range: {single_df['unit_sales'].min()} to {single_df['unit_sales'].max()}")
    
    # Test the core function
    print("\n🔄 Testing generate_growth_rate_store_sku_feature...")
    
    try:
        result = generate_growth_rate_store_sku_feature(
            single_df,
            window_size=7,
            calendar_aligned=True,
            log_level="DEBUG"  # Verbose logging
        )
        
        if result is not None and not result.empty:
            print(f"✅ Success! Generated {len(result)} feature rows")
            print(f"📊 Result columns: {list(result.columns)}")
            print(f"📊 Sample data:")
            print(result.head(2))
            return result
        else:
            print("❌ Function returned empty or None result")
            return None
            
    except Exception as e:
        print(f"❌ Error in generate_growth_rate_store_sku_feature: {e}")
        import traceback
        traceback.print_exc()
        return None


def test_multiple_combinations(df: pd.DataFrame, max_combinations: int = 5):
    """Test with multiple store-item combinations."""
    print(f"\n🧪 MULTIPLE COMBINATIONS TEST (max {max_combinations})")
    print("="*50)
    
    # Get first few combinations
    combinations = df[['store', 'item']].drop_duplicates().head(max_combinations)
    
    results = []
    for idx, (_, row) in enumerate(combinations.iterrows()):
        store, item = row['store'], row['item']
        print(f"\n🔄 Testing {idx+1}/{len(combinations)}: store {store}, item {item}")
        
        # Extract data
        mask = (df['store'] == store) & (df['item'] == item)
        combo_df = df[mask].copy()
        
        if len(combo_df) == 0:
            print(f"   ❌ No data found")
            continue
        
        try:
            result = generate_growth_rate_store_sku_feature(
                combo_df,
                window_size=7,
                calendar_aligned=True,
                log_level="WARNING"  # Reduce noise
            )
            
            if result is not None and not result.empty:
                print(f"   ✅ Generated {len(result)} rows")
                results.append(result)
            else:
                print(f"   ❌ Empty result")
                
        except Exception as e:
            print(f"   ❌ Error: {e}")
    
    print(f"\n📊 Summary: {len(results)} successful combinations out of {len(combinations)}")
    return results


def test_high_level_function(df: pd.DataFrame):
    """Test the high-level function with debug logging."""
    print(f"\n🧪 HIGH-LEVEL FUNCTION TEST")
    print("="*50)
    
    # Enable debug logging
    logging.basicConfig(level=logging.DEBUG, format='%(levelname)s: %(message)s')
    
    print("🔄 Testing generate_growth_rate_features...")
    
    try:
        result = generate_growth_rate_features(
            df=df,
            window_size=7,
            calendar_aligned=True,
            n_jobs=1,  # Single-threaded for easier debugging
            log_level="DEBUG"
        )
        
        if result is not None and not result.empty:
            print(f"✅ Success! Generated {len(result)} total feature rows")
            print(f"📊 Result shape: {result.shape}")
            print(f"📊 Result columns: {list(result.columns)}")
            return result
        else:
            print("❌ High-level function returned empty or None result")
            return None
            
    except Exception as e:
        print(f"❌ Error in generate_growth_rate_features: {e}")
        import traceback
        traceback.print_exc()
        return None


def create_minimal_test_data():
    """Create minimal test data that should work."""
    print("🔧 Creating minimal test data...")
    
    dates = pd.date_range('2023-01-01', periods=30, freq='D')
    
    data = []
    for store in [1, 2]:
        for item in [1001, 1002]:
            for date in dates:
                data.append({
                    'store': store,
                    'item': item,
                    'date': date,
                    'unit_sales': np.random.poisson(5) + 1,  # Ensure non-zero sales
                    'onpromotion': np.random.choice([0, 1], p=[0.8, 0.2]),
                    'weight': 1.0
                })
    
    df = pd.DataFrame(data)
    print(f"✅ Created test data: {len(df)} rows, {len(df[['store', 'item']].drop_duplicates())} combinations")
    return df


def main():
    """Main debug function."""
    print("🐛 GROWTH RATE FEATURES DEBUG TOOL")
    print("="*60)
    
    # Option 1: Test with your actual data
    # Uncomment and modify this section to use your real data
    """
    print("📂 Loading your data...")
    df = pd.read_csv("your_data_file.csv")  # Replace with your data file
    df['date'] = pd.to_datetime(df['date'])
    """
    
    # Option 2: Test with minimal synthetic data
    print("📂 Using synthetic test data...")
    df = create_minimal_test_data()
    
    # Debug data structure
    if not debug_data_structure(df):
        print("❌ Data structure issues found. Please fix before proceeding.")
        return
    
    # Test single store-item
    single_result = test_single_store_item(df)
    
    # Test multiple combinations
    multiple_results = test_multiple_combinations(df)
    
    # Test high-level function
    high_level_result = test_high_level_function(df)
    
    # Summary
    print(f"\n🎯 DEBUG SUMMARY")
    print("="*40)
    print(f"✅ Single store-item test: {'PASS' if single_result is not None else 'FAIL'}")
    print(f"✅ Multiple combinations test: {'PASS' if multiple_results else 'FAIL'}")
    print(f"✅ High-level function test: {'PASS' if high_level_result is not None else 'FAIL'}")
    
    if high_level_result is not None:
        print(f"🎉 SUCCESS! The function works correctly.")
    else:
        print(f"🔧 DEBUGGING NEEDED:")
        print(f"   1. Check your data format and column names")
        print(f"   2. Ensure you have sufficient data per store-item combination")
        print(f"   3. Check for data quality issues (nulls, zeros, etc.)")
        print(f"   4. Try with a smaller subset of your data first")


if __name__ == "__main__":
    main()
