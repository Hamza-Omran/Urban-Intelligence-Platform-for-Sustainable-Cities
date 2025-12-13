"""
Test Factor Analysis Setup
Run this BEFORE running the main factor analysis
"""

import pandas as pd
import os
import sys

print("="*70)
print("TESTING FACTOR ANALYSIS SETUP")
print("="*70)

# Test 1: Check simulation results exist
print("\n[Test 1] Checking simulation results...")
sim_path = "Data/gold/simulation_results.csv"

if os.path.exists(sim_path):
    df = pd.read_csv(sim_path)
    print(f"  Simulation results found")
    print(f"  Records: {len(df):,}")
    print(f"  Columns: {len(df.columns)}")
else:
    print(f"  ERROR: Simulation results not found!")
    print(f"  Expected location: {sim_path}")
    print(f"  Please run Member 5's monte_carlo_simulation.py first")
    sys.exit(1)

# Test 2: Check required features
print("\n[Test 2] Checking required features...")
weather_features = ['temperature_c', 'humidity', 'rain_mm', 
                   'wind_speed_kmh', 'visibility_m']
traffic_features = ['vehicle_count', 'avg_speed_kmh', 'accident_count']
all_required = weather_features + traffic_features

missing = [col for col in all_required if col not in df.columns]
if missing:
    print(f"  ERROR: Missing columns: {missing}")
    sys.exit(1)
else:
    print(f"  All required features present")

# Test 3: Check data quality
print("\n[Test 3] Checking data quality...")
print(f"  Null values: {df[all_required].isnull().sum().sum()}")
print(f"  Feature ranges:")
for col in all_required:
    print(f"    {col:25s}: {df[col].min():.2f} to {df[col].max():.2f}")

# Test 4: Check output directory
print("\n[Test 4] Checking output directory...")
if not os.path.exists("Data/gold"):
    os.makedirs("Data/gold")
    print("  Created: Data/gold/")
else:
    print("  Data/gold/ exists")

# Test 5: Check required packages
print("\n[Test 5] Checking required packages...")
try:
    import numpy
    import matplotlib
    import seaborn
    from sklearn.decomposition import PCA, FactorAnalysis
    from sklearn.preprocessing import StandardScaler
    print("  All packages installed")
except ImportError as e:
    print(f"  ERROR: Missing package: {e}")
    print("  Run: pip install pandas numpy matplotlib seaborn scikit-learn")
    sys.exit(1)

# Summary
print("\n" + "="*70)
print("ALL TESTS PASSED!")
print("="*70)
print("\nYou can now run the factor analysis:")
print("  python Scripts/factor_analysis.py")
print("="*70)