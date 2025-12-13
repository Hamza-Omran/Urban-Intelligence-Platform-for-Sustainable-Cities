import pandas as pd
import numpy as np
import random
from datetime import datetime, timedelta

# ---------------------------------------
# Generate Synthetic Weather Dataset
# ---------------------------------------

num_rows = 5000

weather_conditions = ["Clear", "Rain", "Fog", "Storm", "Snow"]
wrong_weather_conditions = ["CLR", "RN", "Unknown", "BAD", None]

# ---------------------------------------
# Helper: Generate random datetime in mixed formats
# ---------------------------------------
def generate_datetime():
    base_time = datetime(2024, 1, 1, 0, 0)
    dt = base_time + timedelta(minutes=random.randint(0, 200*24*60))

    formats = [
        "%Y-%m-%d %H:%M",
        "%d/%m/%Y %I%p",
        "%Y-%m-%dT%H:%MZ"
    ]

    formatted = dt.strftime(random.choice(formats))

    # Inject wrong date formats
    if random.random() < 0.01:
        formatted = "2099-13-40 25:61"   # impossible date
    if random.random() < 0.01:
        formatted = "Unknown"
    if random.random() < 0.01:
        return None

    return formatted

# ---------------------------------------
# Helper: Generate season from date
# Note: some NULL/invalid season injected later
# ---------------------------------------
def infer_season(dt):
    if pd.isna(dt):
        return None
    try:
        d = pd.to_datetime(dt, errors="coerce")
        if d is None:
            return None
        month = d.month
        if month in [12, 1, 2]:
            return "Winter"
        elif month in [3, 4, 5]:
            return "Spring"
        elif month in [6, 7, 8]:
            return "Summer"
        else:
            return "Autumn"
    except:
        return None


# ---------------------------------------
# Generate base dataframe
# ---------------------------------------
date_col = [generate_datetime() for _ in range(num_rows)]

df = pd.DataFrame({
    "weather_id": np.arange(5001, 5001 + num_rows),
    "date_time": date_col,
    "city": ["London"] * num_rows,
    "season": [infer_season(x) for x in date_col],
    "temperature_c": np.random.uniform(-5, 35, num_rows).round(2),
    "humidity": np.random.randint(20, 100, num_rows),
    "rain_mm": np.random.uniform(0, 50, num_rows).round(2),
    "wind_speed_kmh": np.random.uniform(0, 80, num_rows).round(2),
    "visibility_m": np.random.randint(50, 10000, num_rows),
    "weather_condition": [random.choice(weather_conditions) for _ in range(num_rows)],
    "air_pressure_hpa": np.random.uniform(950, 1050, num_rows).round(2)
})

# ---------------------------------------
# Inject Missing Values
# ---------------------------------------
for col in df.columns:
    df.loc[df.sample(25).index, col] = None

# ---------------------------------------
# Inject Outliers
# ---------------------------------------
df.loc[df.sample(20).index, "temperature_c"] = np.random.choice([-30, -25, 50, 60])
df.loc[df.sample(20).index, "humidity"] = np.random.choice([-10, 150])
df.loc[df.sample(15).index, "rain_mm"] = np.random.uniform(60, 150, 15)
df.loc[df.sample(10).index, "wind_speed_kmh"] = np.random.uniform(100, 250, 10).round(2)
df.loc[df.sample(10).index, "visibility_m"] = 50000
df.loc[df.sample(20).index, "air_pressure_hpa"] = np.random.uniform(800, 1200, 20).round(2)

# Wrong categories
df.loc[df.sample(40).index, "weather_condition"] = [
    random.choice(wrong_weather_conditions) for _ in range(40)
]

# Bad season values
df.loc[df.sample(20).index, "season"] = random.choice(
    ["HOT", "COLD", "BAD", None]
)

# Duplicate rows
df = pd.concat([df, df.sample(50)], ignore_index=True)


# ---------------------------------------
# Save CSV File
# ---------------------------------------
df.to_csv("./Data/weather_raw.csv", index=False)
print("Weather dataset generated")
