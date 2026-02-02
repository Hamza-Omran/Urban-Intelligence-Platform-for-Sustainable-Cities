import pandas as pd
import os
import sys
print("="*70)
print("TESTING MONTE CARLO SIMULATION SETUP")
print("="*70)
print("\n[Test 1] Checking merged dataset...")
merged_path = "output/merged_with_features.parquet"
if os.path.exists(merged_path):
    df = pd.read_parquet(merged_path)
    print(f" Merged dataset found")
    print(f"  Records: {len(df):,}")
    print(f"  Columns: {len(df.columns)}")
else:
    print(f" ERROR: Merged dataset not found!")
    print(f"  Expected location: {merged_path}")
    print(f"  Please run Member 4's data_merging.py first")
    sys.exit(1)
print("\n[Test 2] Checking required columns...")
required_cols = [
    'weather_severity_index',
    'traffic_intensity_score',
    'temperature_c',
    'rain_mm',
    'wind_speed_kmh',
    'visibility_m',
    'humidity',
    'avg_speed_kmh',
    'congestion_level',
    'road_condition',
    'area',
    'hour',
    'date_time'
]
missing = [col for col in required_cols if col not in df.columns]
if missing:
    print(f" ERROR: Missing columns: {missing}")
    sys.exit(1)
else:
    print(f" All required columns present")
print("\n[Test 3] Checking data quality...")
print(f"  Null values: {df.isnull().sum().sum()}")
print(f"  Weather severity range: {df['weather_severity_index'].min():.2f} - {df['weather_severity_index'].max():.2f}")
print(f"  Traffic intensity range: {df['traffic_intensity_score'].min():.2f} - {df['traffic_intensity_score'].max():.2f}")
print("\n[Test 4] Checking output directories...")
if not os.path.exists("Data/gold"):
    os.makedirs("Data/gold")
    print("  Created: Data/gold/")
else:
    print("   Data/gold/ exists")
if not os.path.exists("output/plots"):
    os.makedirs("output/plots")
    print("  Created: output/plots/")
else:
    print("   output/plots/ exists")
print("\n[Test 5] Checking required packages...")
try:
    import numpy
    import matplotlib
    import seaborn
    print("   All packages installed")
except ImportError as e:
    print(f"   Missing package: {e}")
    print("  Run: pip install pandas numpy matplotlib seaborn pyarrow")
    sys.exit(1)
print("\n" + "="*70)
print(" ALL TESTS PASSED!")
print("="*70)
print("\nYou can now run the main simulation:")
print("  python Scripts/monte_carlo_simulation.py")
print("="*70)
