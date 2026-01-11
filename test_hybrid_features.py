"""
Test script to diagnose phase4_validation.ipynb issues
"""
import sys
sys.path.insert(0, r'c:\Users\ibito\Documents\Final_Year_Project')

import warnings
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.preprocessing import LabelEncoder

print("="*80)
print("TESTING HYBRID MODEL PREDICTION SETUP")
print("="*80)

# Load test data
hybrid_test = pd.read_csv('project_data/train_test_split/hybrid/test.csv')
print(f"\n✓ Loaded hybrid test data: {hybrid_test.shape}")
print(f"  Columns: {list(hybrid_test.columns[:10])}...")

# Check for required features
print("\n🔍 Checking for required features...")

# Phase3 training features
expected_temporal = [
    'Temperature_C', 'Rainfall_mm', 'Humidity_percent', 'CO2_ppm',
    'GDD', 'Cumulative_Rainfall', 'Days_Into_Season',
    'Is_Rainy_Season', 'Is_Peak_Growing',
    'Heat_Stress', 'Cold_Stress', 'Rainfall_Anomaly',
    'Drought_Risk', 'Flood_Risk',
    'Yield_Lag_1', 'Yield_MA_3yr', 'Yield_YoY_Change'
]

expected_static = [
    'Avg_pH', 'Avg_Nitrogen_ppm', 'Avg_Phosphorus_ppm', 'Avg_Organic_Matter_Percent',
    'pH_Temperature_Interaction', 'Nitrogen_Rainfall_Interaction',
    'Yield_Lag_2', 'Yield_Lag_3', 'Temp_MA_3yr', 'Rain_MA_3yr',
    'Temp_YoY_Change', 'Rain_YoY_Change', 'Yield_Volatility_3yr'
]

# Check temporal
missing_temporal = [col for col in expected_temporal if col not in hybrid_test.columns]
available_temporal = [col for col in expected_temporal if col in hybrid_test.columns]

print(f"\n  Temporal features:")
print(f"    Expected: {len(expected_temporal)}")
print(f"    Available: {len(available_temporal)}")
if missing_temporal:
    print(f"    ❌ Missing ({len(missing_temporal)}): {missing_temporal[:5]}...")
else:
    print(f"    ✅ All present!")

# Check static
missing_static = [col for col in expected_static if col not in hybrid_test.columns]
available_static = [col for col in expected_static if col in hybrid_test.columns]

print(f"\n  Static features:")
print(f"    Expected: {len(expected_static)}")
print(f"    Available: {len(available_static)}")
if missing_static:
    print(f"    ❌ Missing ({len(missing_static)}): {missing_static[:5]}...")
else:
    print(f"    ✅ All present!")

# Check for Region/Zone column
print(f"\n  Categorical features:")
if 'Region' in hybrid_test.columns:
    print(f"    ✅ Region column present")
    print(f"       Unique values: {hybrid_test['Region'].nunique()}")
elif 'Zone' in hybrid_test.columns:
    print(f"    ✅ Zone column present")
    print(f"       Unique values: {hybrid_test['Zone'].nunique()}")
else:
    print(f"    ❌ Neither Region nor Zone column found!")

if 'Crop' in hybrid_test.columns:
    print(f"    ✅ Crop column present")
    print(f"       Unique values: {hybrid_test['Crop'].nunique()}")
    print(f"       Crops: {list(hybrid_test['Crop'].unique())}")
else:
    print(f"    ❌ Crop column not found!")

print("\n" + "="*80)
print("DIAGNOSIS COMPLETE")
print("="*80)
