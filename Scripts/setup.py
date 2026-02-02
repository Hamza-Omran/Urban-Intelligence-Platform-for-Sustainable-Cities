import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')
np.random.seed(42)
MERGED_DATA_PATH = r"output/merged_with_features.parquet"
SIMULATION_RESULTS_PATH = r"output/simulation_results.csv"
GOLD_BUCKET_PATH = r"Data/gold"
import os
os.makedirs(GOLD_BUCKET_PATH, exist_ok=True)
os.makedirs("output/plots", exist_ok=True)
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
