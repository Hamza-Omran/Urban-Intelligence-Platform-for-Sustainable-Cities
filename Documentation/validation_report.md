# HDFS Data Validation Report

**Date:** 2025-12-11 22:07:48

## Validation Log

```
[2025-12-11 22:07:33] ============================================================
[2025-12-11 22:07:33] HDFS Data Validation
[2025-12-11 22:07:33] ============================================================
[2025-12-11 22:07:33] Validating HDFS directory structure...
[2025-12-11 22:07:34]   Directory exists: /user/bigdata/weather
[2025-12-11 22:07:36]   Directory exists: /user/bigdata/traffic
[2025-12-11 22:07:37]   Directory exists: /user/bigdata/cleaned
[2025-12-11 22:07:37] 
============================================================
[2025-12-11 22:07:37] File Integrity Validation
[2025-12-11 22:07:37] ============================================================
[2025-12-11 22:07:37] 
Validating Weather data (weather dir)...
[2025-12-11 22:07:38]   File size: 196,296 bytes
[2025-12-11 22:07:40]   Parquet file valid
[2025-12-11 22:07:40]   Rows: 4,794
[2025-12-11 22:07:40]   Columns: 11
[2025-12-11 22:07:40]   Columns: weather_id, date_time, city, season, temperature_c, humidity, rain_mm, wind_speed_kmh, visibility_m, weather_condition, air_pressure_hpa
[2025-12-11 22:07:40] 
Validating Traffic data (traffic dir)...
[2025-12-11 22:07:41]   File size: 154,296 bytes
[2025-12-11 22:07:43]   Parquet file valid
[2025-12-11 22:07:43]   Rows: 4,813
[2025-12-11 22:07:43]   Columns: 10
[2025-12-11 22:07:43]   Columns: traffic_id, date_time, city, area, vehicle_count, avg_speed_kmh, accident_count, congestion_level, road_condition, visibility_m
[2025-12-11 22:07:43] 
Validating Weather data (cleaned dir)...
[2025-12-11 22:07:44]   File size: 196,296 bytes
[2025-12-11 22:07:46]   Parquet file valid
[2025-12-11 22:07:46]   Rows: 4,794
[2025-12-11 22:07:46]   Columns: 11
[2025-12-11 22:07:46]   Columns: weather_id, date_time, city, season, temperature_c, humidity, rain_mm, wind_speed_kmh, visibility_m, weather_condition, air_pressure_hpa
[2025-12-11 22:07:46] 
Validating Traffic data (cleaned dir)...
[2025-12-11 22:07:47]   File size: 154,296 bytes
[2025-12-11 22:07:48]   Parquet file valid
[2025-12-11 22:07:48]   Rows: 4,813
[2025-12-11 22:07:48]   Columns: 10
[2025-12-11 22:07:48]   Columns: traffic_id, date_time, city, area, vehicle_count, avg_speed_kmh, accident_count, congestion_level, road_condition, visibility_m
[2025-12-11 22:07:48] 
============================================================
[2025-12-11 22:07:48] Validation Summary
[2025-12-11 22:07:48] ============================================================
```
