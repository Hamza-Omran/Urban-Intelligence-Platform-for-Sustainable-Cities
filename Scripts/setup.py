"""
Member 5 – Monte Carlo Simulation for Traffic Risk Prediction
------------------------------------------------------------

This script performs:
1. Loading merged dataset from Member 4
2. Defining simulation scenarios (weather extremes)
3. Running Monte Carlo simulations (10,000+ iterations)
4. Computing congestion and accident probabilities
5. Generating probability distributions
6. Saving results to Gold layer
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# Set random seed for reproducibility
np.random.seed(42)

# ================================================================
# FILE PATHS
# ================================================================
MERGED_DATA_PATH = r"output/merged_with_features.parquet"
SIMULATION_RESULTS_PATH = r"output/simulation_results.csv"
GOLD_BUCKET_PATH = r"Data/gold"  # Will upload to MinIO gold bucket

# Create output directories
import os
os.makedirs(GOLD_BUCKET_PATH, exist_ok=True)
os.makedirs("output/plots", exist_ok=True)

# ================================================================
# STEP 1 — Load Merged Dataset
# ================================================================
print("="*60)
print("Loading merged dataset from Member 4...")
print("="*60)

df = pd.read_parquet(MERGED_DATA_PATH)

print(f"\nDataset loaded successfully!")
print(f"Total records: {len(df)}")
print(f"\nColumns available:")
print(df.columns.tolist())
print(f"\nDataset preview:")
print(df.head())
print(f"\nDataset statistics:")
print(df.describe())