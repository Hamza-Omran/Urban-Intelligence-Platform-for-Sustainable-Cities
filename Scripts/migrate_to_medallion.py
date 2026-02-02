import os
from datetime import datetime
from hdfs_utils import (
    create_hdfs_directory,
    upload_to_hdfs,
    list_hdfs_directory,
    get_hdfs_file_size,
    run_hdfs_command
)
from config import (
    HDFS_BRONZE_PATH,
    HDFS_BRONZE_WEATHER_PATH,
    HDFS_BRONZE_TRAFFIC_PATH,
    HDFS_SILVER_PATH,
    HDFS_SILVER_WEATHER_PATH,
    HDFS_SILVER_TRAFFIC_PATH,
    HDFS_GOLD_PATH,
    WEATHER_RAW_PATH,
    TRAFFIC_RAW_PATH,
    WEATHER_CLEANED_PATH,
    TRAFFIC_CLEANED_PATH,
    HDFS_BIN
)
def log_message(message):
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"[{timestamp}] {message}")
def create_medallion_structure():
    log_message("Creating Medallion Architecture directory structure...")
    directories = [
        HDFS_BRONZE_PATH,
        HDFS_BRONZE_WEATHER_PATH,
        HDFS_BRONZE_TRAFFIC_PATH,
        HDFS_SILVER_PATH,
        HDFS_SILVER_WEATHER_PATH,
        HDFS_SILVER_TRAFFIC_PATH,
        HDFS_GOLD_PATH
    ]
    for directory in directories:
        log_message(f"  Creating: {directory}")
        create_hdfs_directory(directory)
    log_message("Directory structure created successfully!\n")
def upload_raw_data_to_bronze():
    log_message("=" * 60)
    log_message("BRONZE LAYER - Uploading Raw Data")
    log_message("=" * 60)
    log_message("\n1. Uploading weather raw data...")
    if os.path.exists(WEATHER_RAW_PATH):
        weather_bronze_path = f"{HDFS_BRONZE_WEATHER_PATH}/weather_raw.csv"
        if upload_to_hdfs(WEATHER_RAW_PATH, weather_bronze_path):
            size = get_hdfs_file_size(weather_bronze_path)
            log_message(f"    Weather raw data uploaded: {size:,} bytes")
            log_message(f"   Location: {weather_bronze_path}")
        else:
            log_message(f"    Failed to upload weather raw data")
    else:
        log_message(f"    Weather raw file not found: {WEATHER_RAW_PATH}")
    log_message("\n2. Uploading traffic raw data...")
    if os.path.exists(TRAFFIC_RAW_PATH):
        traffic_bronze_path = f"{HDFS_BRONZE_TRAFFIC_PATH}/traffic_raw.csv"
        if upload_to_hdfs(TRAFFIC_RAW_PATH, traffic_bronze_path):
            size = get_hdfs_file_size(traffic_bronze_path)
            log_message(f"    Traffic raw data uploaded: {size:,} bytes")
            log_message(f"   Location: {traffic_bronze_path}")
        else:
            log_message(f"    Failed to upload traffic raw data")
    else:
        log_message(f"    Traffic raw file not found: {TRAFFIC_RAW_PATH}")
def upload_cleaned_data_to_silver():
    log_message("\n" + "=" * 60)
    log_message("SILVER LAYER - Uploading Cleaned Data")
    log_message("=" * 60)
    log_message("\n1. Uploading weather cleaned data...")
    if os.path.exists(WEATHER_CLEANED_PATH):
        weather_silver_path = f"{HDFS_SILVER_WEATHER_PATH}/weather_cleaned.parquet"
        if upload_to_hdfs(WEATHER_CLEANED_PATH, weather_silver_path):
            size = get_hdfs_file_size(weather_silver_path)
            log_message(f"    Weather cleaned data uploaded: {size:,} bytes")
            log_message(f"   Location: {weather_silver_path}")
        else:
            log_message(f"    Failed to upload weather cleaned data")
    else:
        log_message(f"    Weather cleaned file not found: {WEATHER_CLEANED_PATH}")
    log_message("\n2. Uploading traffic cleaned data...")
    if os.path.exists(TRAFFIC_CLEANED_PATH):
        traffic_silver_path = f"{HDFS_SILVER_TRAFFIC_PATH}/traffic_cleaned.parquet"
        if upload_to_hdfs(TRAFFIC_CLEANED_PATH, traffic_silver_path):
            size = get_hdfs_file_size(traffic_silver_path)
            log_message(f"    Traffic cleaned data uploaded: {size:,} bytes")
            log_message(f"   Location: {traffic_silver_path}")
        else:
            log_message(f"    Failed to upload traffic cleaned data")
    else:
        log_message(f"    Traffic cleaned file not found: {TRAFFIC_CLEANED_PATH}")
def remove_old_structure():
    log_message("\n" + "=" * 60)
    log_message("CLEANUP - Removing Old Structure")
    log_message("=" * 60)
    old_directories = [
        "/user/bigdata/weather",
        "/user/bigdata/traffic",
        "/user/bigdata/cleaned"
    ]
    for old_dir in old_directories:
        log_message(f"\nRemoving: {old_dir}")
        command = [
            HDFS_BIN,
            "dfs", "-rm", "-r", "-skipTrash",
            old_dir
        ]
        success, output, error = run_hdfs_command(command)
        if success:
            log_message(f"    Removed: {old_dir}")
        else:
            if "No such file or directory" in error:
                log_message(f"   - Already removed: {old_dir}")
            else:
                log_message(f"    Failed to remove: {error}")
def verify_medallion_structure():
    log_message("\n" + "=" * 60)
    log_message("VERIFICATION - Medallion Architecture")
    log_message("=" * 60)
    layers = {
        "BRONZE (Raw Data)": HDFS_BRONZE_PATH,
        "SILVER (Cleaned Data)": HDFS_SILVER_PATH,
        "GOLD (Aggregated Data)": HDFS_GOLD_PATH
    }
    for layer_name, layer_path in layers.items():
        log_message(f"\n{layer_name}: {layer_path}")
        files = list_hdfs_directory(layer_path)
        if files:
            subdirs = [f.split('/')[-1] for f in files]
            log_message(f"  Subdirectories: {', '.join(subdirs)}")
            for subdir_full_path in files:
                subdir_files = list_hdfs_directory(subdir_full_path)
                if subdir_files:
                    for file_path in subdir_files:
                        file_name = file_path.split('/')[-1]
                        file_size = get_hdfs_file_size(file_path)
                        log_message(f"    - {file_path}: {file_size:,} bytes")
        else:
            log_message(f"  (empty)")
def main():
    log_message("=" * 60)
    log_message("HDFS Migration to Medallion Architecture")
    log_message("=" * 60)
    log_message("Implementing Bronze → Silver → Gold structure\n")
    create_medallion_structure()
    upload_raw_data_to_bronze()
    upload_cleaned_data_to_silver()
    log_message("\nDo you want to remove the old directory structure?")
    log_message("(Old: /user/bigdata/weather, /user/bigdata/traffic, /user/bigdata/cleaned)")
    response = input("Enter 'yes' to remove old structure: ")
    if response.lower() == 'yes':
        remove_old_structure()
    else:
        log_message("Skipping removal of old structure")
    verify_medallion_structure()
    log_message("\n" + "=" * 60)
    log_message("Migration Complete!")
    log_message("=" * 60)
    log_message("\nYour data is now organized as:")
    log_message("   Bronze - Raw CSV files (as ingested)")
    log_message("   Silver - Cleaned Parquet files (transformed)")
    log_message("   Gold - Aggregated data (business-ready)")
if __name__ == "__main__":
    main()
