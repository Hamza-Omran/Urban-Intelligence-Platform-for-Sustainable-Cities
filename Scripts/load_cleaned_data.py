import pandas as pd

weather_path = r"Data\weather_cleaned.parquet"
traffic_path = r"Data/traffic_cleaned.parquet"

weather = pd.read_parquet(weather_path)
traffic = pd.read_parquet(traffic_path)

print("Weather dataset:")
print(weather.head())
print(weather.info())

print("\nTraffic dataset:")
print(traffic.head())
print(traffic.info())
