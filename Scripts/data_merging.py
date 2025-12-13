"""
Member 4 – Data Integration, Merging & Feature Engineering
----------------------------------------------------------

This script performs:
1. Loading cleaned datasets
2. Timestamp normalization + timezone alignment
3. Exact merge on ["city", "date_time"]
4. Conflict resolution
5. Feature engineering on merged dataset
6. Saving merged dataset (Parquet)
7. Outputting merge + feature engineering validation report

Requirements:
    pip install pandas pyarrow numpy pytz
"""

import pandas as pd
import numpy as np
from datetime import datetime
import pytz

# ================================================================
# FILE PATHS
# ================================================================
WEATHER_PATH = r"Data\weather_cleaned.parquet"
TRAFFIC_PATH = r"Data/traffic_cleaned.parquet"
MERGED_OUTPUT = r"output\merged_dataset.parquet"
FEATURED_OUTPUT = r"output/merged_with_features.parquet"
REPORT_PATH = r"output\merge_validation_report.txt"

# ================================================================
# STEP 1 — Load cleaned data
# ================================================================
print("Loading cleaned datasets...")
weather = pd.read_parquet(WEATHER_PATH)
traffic = pd.read_parquet(TRAFFIC_PATH)

# ================================================================
# STEP 2 — Standardize timestamps & timezone alignment
# ================================================================
print("Aligning timestamps and converting to UTC...")

LOCAL_TZ = pytz.timezone("Europe/London")

def to_utc(series):
    """Convert naive timestamps → timezone-aware UTC."""
    return series.apply(lambda dt: LOCAL_TZ.localize(dt).astimezone(pytz.utc))

weather["date_time_utc"] = to_utc(weather["date_time"])
traffic["date_time_utc"] = to_utc(traffic["date_time"])

# Normalize city formatting
weather["city"] = weather["city"].astype(str).str.strip().str.lower()
traffic["city"] = traffic["city"].astype(str).str.strip().str.lower()

# ================================================================
# STEP 3 — EXACT MERGE
# ================================================================
print("Performing exact merge...")

merged = pd.merge(
    weather,
    traffic,
    on=["city", "date_time"],
    how="inner",
    suffixes=("_weather", "_traffic")
)

# ================================================================
# STEP 4 — Conflict Resolution
# ================================================================
print("Resolving conflicts...")

# Both datasets contain visibility_m → merge them using the mean
if "visibility_m_weather" in merged.columns and "visibility_m_traffic" in merged.columns:
    merged["visibility_m"] = merged[
        ["visibility_m_weather", "visibility_m_traffic"]
    ].mean(axis=1)

    merged.drop(columns=["visibility_m_weather", "visibility_m_traffic"], inplace=True)

# Save intermediate merged dataset (required by project)
merged.to_parquet(MERGED_OUTPUT, index=False)

# ================================================================
# STEP 5 — FEATURE ENGINEERING
# ================================================================
print("Applying feature engineering...")

df = merged.copy()

# --------------------
# Time-based features
# --------------------
df["hour"] = df["date_time"].dt.hour
df["day_of_week"] = df["date_time"].dt.dayofweek  # Monday = 0
df["is_weekend"] = df["day_of_week"].isin([5, 6]).astype(int)

# --------------------
# Weather Severity Index
# --------------------
def compute_weather_severity(row):
    """Weighted severity formula."""
    temp = abs(row["temperature_c"] - 15)        # deviation from pleasant 15°C
    rain = row["rain_mm"]
    wind = row["wind_speed_kmh"]
    vis = 1 / (row["visibility_m"] + 1e-6)       # lower visibility = worse

    score = (
        0.30 * temp +
        0.30 * rain +
        0.25 * wind +
        0.15 * vis
    )

    return score

df["weather_severity_index"] = df.apply(compute_weather_severity, axis=1)

# --------------------
# Traffic Intensity Score
# --------------------
def compute_traffic_intensity(row):
    inv_speed = 1 / (row["avg_speed_kmh"] + 1e-6)   # lower speed = worse
    veh = row["vehicle_count"]
    acc = row["accident_count"]

    score = (
        0.40 * veh +
        0.35 * inv_speed +
        0.25 * acc
    )
    return score

df["traffic_intensity_score"] = df.apply(compute_traffic_intensity, axis=1)

# Save final dataset
df.to_parquet(FEATURED_OUTPUT, index=False)

# ================================================================
# STEP 6 — VALIDATION REPORT
# ================================================================
print("Generating validation report...")

report = {
    "Weather rows": len(weather),
    "Traffic rows": len(traffic),
    "Merged rows (exact match)": len(merged),
    "Missing from weather": len(weather) - len(merged),
    "Missing from traffic": len(traffic) - len(merged),
    "Merge rate (%)": round(len(merged) / min(len(weather), len(traffic)) * 100, 2),
    "Final dataset with features": len(df),
}

with open(REPORT_PATH, "w") as f:
    f.write("=== MEMBER 4 – Merge + Feature Engineering Report ===\n\n")
    for k, v in report.items():
        f.write(f"{k}: {v}\n")

print("\nStep 3 completed successfully!")
print(f"Final dataset saved to: {FEATURED_OUTPUT}")
