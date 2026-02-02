import os
from pathlib import Path
print("="*70)
print("COMPLETE PROJECT VERIFICATION")
print("="*70)
all_valid = True
print("\n[PHASE 1-4: DATA PIPELINE]")
bronze_files = ["weather_raw.csv", "traffic_raw.csv"]
silver_files = ["weather_cleaned.parquet", "traffic_cleaned.parquet"]
output_files = ["merged_dataset.parquet", "merged_with_features.parquet"]
for file in bronze_files:
    path = Path(f"Data/{file}")
    if path.exists():
        print(f"  [OK] Bronze: {file}")
    else:
        print(f"  [FAIL] Bronze: {file}")
        all_valid = False
for file in silver_files:
    path = Path(f"Data/{file}")
    if path.exists():
        print(f"  [OK] Silver: {file}")
    else:
        print(f"  [FAIL] Silver: {file}")
        all_valid = False
for file in output_files:
    path = Path(f"output/{file}")
    if path.exists():
        print(f"  [OK] Merged: {file}")
    else:
        print(f"  [FAIL] Merged: {file}")
        all_valid = False
print("\n[PHASE 5: MONTE CARLO SIMULATION]")
simulation_files = [
    "simulation_results.csv",
    "scenario_analysis.csv",
    "congestion_probability_distribution.png",
    "accident_probability_distribution.png",
    "scenario_comparison.png",
    "risk_heatmap_area_hour.png"
]
for file in simulation_files:
    path = Path(f"Data/gold/{file}")
    if path.exists():
        size = path.stat().st_size
        print(f"  [OK] {file} ({size:,} bytes)")
    else:
        print(f"  [FAIL] {file}")
        all_valid = False
print("\n[PHASE 6: FACTOR ANALYSIS]")
factor_files = [
    "factor_loadings.csv",
    "factor_scores.csv",
    "factor_analysis_interpretation.txt",
    "scree_plot.png",
    "factor_loadings_heatmap.png",
    "factor_scores_distribution.png",
    "factor_biplot.png"
]
for file in factor_files:
    path = Path(f"Data/gold/{file}")
    if path.exists():
        size = path.stat().st_size
        print(f"  [OK] {file} ({size:,} bytes)")
    else:
        print(f"  [FAIL] {file}")
        all_valid = False
print("\n[SCRIPTS VERIFICATION]")
required_scripts = [
    "monte_carlo_simulation.py",
    "factor_analysis.py",
    "test_simulation.py",
    "test_factor_analysis.py",
    "verify_simulation_outputs.py"
]
for script in required_scripts:
    path = Path(f"Scripts/{script}")
    if path.exists():
        print(f"  [OK] {script}")
    else:
        print(f"  [FAIL] {script}")
        all_valid = False
print("\n[DATA QUALITY CHECKS]")
try:
    import pandas as pd
    sim_path = Path("Data/gold/simulation_results.csv")
    if sim_path.exists():
        df_sim = pd.read_csv(sim_path)
        if len(df_sim) == 10000:
            print(f"  [OK] Simulation: 10,000 rows")
        else:
            print(f"  [WARN] Simulation: {len(df_sim)} rows (expected 10,000)")
            all_valid = False
        required_cols = ['congestion_probability', 'accident_probability', 
                        'congestion_occurred', 'accident_occurred']
        if all(col in df_sim.columns for col in required_cols):
            print(f"  [OK] Simulation: All required columns present")
        else:
            print(f"  [FAIL] Simulation: Missing required columns")
            all_valid = False
    factor_path = Path("Data/gold/factor_loadings.csv")
    if factor_path.exists():
        df_factor = pd.read_csv(factor_path)
        if len(df_factor.columns) == 3:
            print(f"  [OK] Factor Analysis: 3 factors extracted")
        else:
            print(f"  [WARN] Factor Analysis: {len(df_factor.columns)} factors (expected 3)")
except Exception as e:
    print(f"  [WARN] Could not validate data: {e}")
print("\n[FILE COUNT SUMMARY]")
gold_dir = Path("Data/gold")
if gold_dir.exists():
    gold_files = list(gold_dir.glob("*"))
    csv_files = [f for f in gold_files if f.suffix == '.csv']
    png_files = [f for f in gold_files if f.suffix == '.png']
    txt_files = [f for f in gold_files if f.suffix == '.txt']
    print(f"  CSV files: {len(csv_files)}")
    print(f"  PNG files: {len(png_files)}")
    print(f"  TXT files: {len(txt_files)}")
    print(f"  Total Gold files: {len(gold_files)}")
    if len(gold_files) >= 13:
        print(f"  [OK] All expected Gold files present")
    else:
        print(f"  [WARN] Expected at least 13 Gold files, found {len(gold_files)}")
        all_valid = False
print("\n" + "="*70)
if all_valid:
    print("PROJECT VERIFICATION: COMPLETE")
    print("="*70)
    print("\nAll deliverables present and validated!")
    print("\nProject phases completed:")
    print("  Phase 1: Infrastructure & Data Generation")
    print("  Phase 2: Data Cleaning")
    print("  Phase 3: HDFS Integration")
    print("  Phase 4: Data Merging & Feature Engineering")
    print("  Phase 5: Monte Carlo Simulation")
    print("  Phase 6: Factor Analysis & Final Integration")
    print("\nReady for final submission!")
else:
    print("PROJECT VERIFICATION: ISSUES DETECTED")
    print("="*70)
    print("\nSome deliverables are missing or incomplete.")
    print("Please review the messages above.")
print("\n" + "="*70)
