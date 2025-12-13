import pandas as pd
import numpy as np
import random
from datetime import datetime, timedelta

# ---------------------------------------
# Generate Synthetic Traffic Dataset
# ---------------------------------------

# Number of rows
num_rows = 5000

# Predefined lists
areas = [
    "Camden", "Chelsea", "Islington", "Kensington",
    "Southwark", "Westminster", "Hammersmith", "Hackney"
]

congestion_levels = ["Low", "Medium", "High"]
wrong_congestion = ["LOWW", "MED", "HIGG", "Unknown", None]

road_conditions = ["Dry", "Wet", "Snowy", "Damaged"]
wrong_road_conditions = ["Bad", "Broken", "Invalid", None]

# ---------------------------------------
#  Generate random datetime in mixed formats
# ---------------------------------------
def generate_datetime():
    base_time = datetime(2024, 1, 1, 0, 0)
    dt = base_time + timedelta(minutes=random.randint(0, 200*24*60))
    
    formats = [
        "%Y-%m-%d %H:%M",
        "%d/%m/%Y %I%p",
        "%Y-%m-%dT%H:%MZ"
    ]
    
    # Randomly choose format
    formatted = dt.strftime(random.choice(formats))

    # Inject wrong format
    if random.random() < 0.01:
        formatted = "2099-00-00 99:99"

    # Inject garbage value
    if random.random() < 0.01:
        formatted = "TBD"

    # Inject NULL
    if random.random() < 0.01:
        return None

    return formatted


# ---------------------------------------
# Create DataFrame
# ---------------------------------------
df = pd.DataFrame({
    "traffic_id": np.arange(9001, 9001 + num_rows),
    "date_time": [generate_datetime() for _ in range(num_rows)],
    "city": ["London"] * num_rows,
    "area": [random.choice(areas) for _ in range(num_rows)],
    "vehicle_count": np.random.randint(0, 5000, num_rows),
    "avg_speed_kmh": np.random.uniform(5, 120, num_rows).round(2),
    "accident_count": np.random.randint(0, 10, num_rows),
    "congestion_level": [random.choice(congestion_levels) for _ in range(num_rows)],
    "road_condition": [random.choice(road_conditions) for _ in range(num_rows)],
    "visibility_m": np.random.randint(200, 15000, num_rows)
})


# ---------------------------------------
# Inject Missing Values
# ---------------------------------------
for col in df.columns:
    df.loc[df.sample(20).index, col] = None


# ---------------------------------------
# Inject Outliers
# ---------------------------------------
df.loc[df.sample(20).index, "vehicle_count"] = np.random.randint(15000, 30000, 20)
df.loc[df.sample(20).index, "avg_speed_kmh"] = -np.random.uniform(5, 40, 20)
df.loc[df.sample(10).index, "visibility_m"] = 50000
df.loc[df.sample(10).index, "accident_count"] = np.random.randint(20, 60, 10)

# Wrong categories
df.loc[df.sample(30).index, "congestion_level"] = [random.choice(wrong_congestion) for _ in range(30)]
df.loc[df.sample(20).index, "road_condition"] = [random.choice(wrong_road_conditions) for _ in range(20)]

# Duplicates
df = pd.concat([df, df.sample(40)], ignore_index=True)


# ---------------------------------------
# Save CSV File
# ---------------------------------------
df.to_csv("./Data/traffic_raw.csv", index=False)
print("Traffic dataset generated")
