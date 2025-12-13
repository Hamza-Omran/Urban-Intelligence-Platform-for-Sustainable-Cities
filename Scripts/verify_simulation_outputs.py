"""
Verify all Monte Carlo simulation outputs
Run this after the main simulation to check everything
"""

import os
import pandas as pd
from pathlib import Path

print("="*70)
print("VERIFYING MONTE CARLO SIMULATION OUTPUTS")
print("="*70)

gold_dir = Path("Data/gold")
results_valid = True

# Expected files
expected_files = {
    "CSV Files": [
        "simulation_results.csv",
        "scenario_analysis.csv"
    ],
    "Visualizations": [
        "congestion_probability_distribution.png",
        "accident_probability_distribution.png",
        "scenario_comparison.png",
        "risk_heatmap_area_hour.png"
    ]
}

# Check each category
for category, files in expected_files.items():
    print(f"\n[{category}]")
    for filename in files:
        filepath = gold_dir / filename
        if filepath.exists():
            size = filepath.stat().st_size
            if size > 0:
                print(f"  [OK] {filename} ({size:,} bytes)")
            else:
                print(f"  [WARN] {filename} (empty file)")
                results_valid = False
        else:
            print(f"  [FAIL] {filename} (not found)")
            results_valid = False

# Validate simulation_results.csv
print("\n" + "="*70)
print("VALIDATING SIMULATION RESULTS")
print("="*70)

results_file = gold_dir / "simulation_results.csv"
if results_file.exists():
    try:
        df = pd.read_csv(results_file)
        print(f"\n[Data Quality Checks]")
        print(f"  Total rows: {len(df):,}")
        print(f"  Expected: 10,000")
        
        if len(df) == 10000:
            print(f"  [OK] Row count correct")
        else:
            print(f"  [WARN] Row count mismatch")
            results_valid = False
        
        # Check required columns
        required_cols = [
            'simulation_id',
            'congestion_probability',
            'accident_probability',
            'congestion_occurred',
            'accident_occurred',
            'active_scenarios'
        ]
        
        missing_cols = [col for col in required_cols if col not in df.columns]
        if missing_cols:
            print(f"  [FAIL] Missing columns: {missing_cols}")
            results_valid = False
        else:
            print(f"  [OK] All required columns present")
        
        # Check value ranges
        print(f"\n[Value Range Checks]")
        
        if df['congestion_probability'].between(0, 1).all():
            print(f"  [OK] Congestion probabilities valid (0-1)")
        else:
            print(f"  [FAIL] Invalid congestion probabilities")
            results_valid = False
        
        if df['accident_probability'].between(0, 1).all():
            print(f"  [OK] Accident probabilities valid (0-1)")
        else:
            print(f"  [FAIL] Invalid accident probabilities")
            results_valid = False
        
        if df['congestion_occurred'].isin([0, 1]).all():
            print(f"  [OK] Congestion occurred values valid (0 or 1)")
        else:
            print(f"  [FAIL] Invalid congestion occurred values")
            results_valid = False
        
        # Summary statistics
        print(f"\n[Summary Statistics]")
        print(f"  Avg Congestion Probability: {df['congestion_probability'].mean():.2%}")
        print(f"  Avg Accident Probability: {df['accident_probability'].mean():.2%}")
        print(f"  Congestion Events: {df['congestion_occurred'].sum():,} ({df['congestion_occurred'].mean():.2%})")
        print(f"  Accident Events: {df['accident_occurred'].sum():,} ({df['accident_occurred'].mean():.2%})")
        
        # Scenario distribution
        print(f"\n[Scenario Distribution]")
        scenario_counts = df['active_scenarios'].value_counts().head(10)
        for scenario, count in scenario_counts.items():
            print(f"  {scenario}: {count:,} ({count/len(df)*100:.1f}%)")
        
    except Exception as e:
        print(f"  [FAIL] Error reading CSV: {e}")
        results_valid = False
else:
    print(f"  [FAIL] simulation_results.csv not found")
    results_valid = False

# Validate scenario_analysis.csv
print("\n" + "="*70)
print("VALIDATING SCENARIO ANALYSIS")
print("="*70)

scenario_file = gold_dir / "scenario_analysis.csv"
if scenario_file.exists():
    try:
        df_scenario = pd.read_csv(scenario_file)
        print(f"\n  Total scenarios analyzed: {len(df_scenario)}")
        print(f"\n  Scenarios:")
        for _, row in df_scenario.iterrows():
            print(f"    - {row['Scenario']}: {row['Occurrences']:,} occurrences")
        print(f"  [OK] Scenario analysis valid")
    except Exception as e:
        print(f"  [FAIL] Error reading scenario CSV: {e}")
        results_valid = False
else:
    print(f"  [FAIL] scenario_analysis.csv not found")
    results_valid = False

# Final verdict
print("\n" + "="*70)
if results_valid:
    print("✅ ALL OUTPUTS VALID - SIMULATION SUCCESSFUL!")
    print("="*70)
    print("\nNext Steps:")
    print("1. Review visualizations in Data/gold/")
    print("2. Upload files to MinIO (if not done automatically)")
    print("3. Hand off simulation_results.csv to Member 6")
else:
    print("⚠️  SOME ISSUES DETECTED - REVIEW ABOVE MESSAGES")
    print("="*70)

print("\nAll files location: Data/gold/")
print("="*70)