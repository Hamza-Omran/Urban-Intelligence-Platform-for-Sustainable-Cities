import os
import time
from datetime import datetime
from minio_utils import get_client, list_bucket_objects
from hdfs_utils import (
    create_hdfs_directory,
    upload_to_hdfs,
    list_hdfs_directory,
    get_hdfs_file_size,
    check_hdfs_running
)
from config import (
    HDFS_WEATHER_PATH,
    HDFS_TRAFFIC_PATH,
    HDFS_CLEANED_PATH,
    TEMP_DIR,
    MINIO_BUCKET_SILVER,
    WEATHER_CLEANED_FILE,
    TRAFFIC_CLEANED_FILE,
    TRANSFER_LOG_FILE,
    ensure_dir
)
LOCAL_TEMP_DIR = TEMP_DIR
TRANSFER_LOG = []
def log_message(message):
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    log_entry = f"[{timestamp}] {message}"
    print(log_entry)
    TRANSFER_LOG.append(log_entry)
def setup_hdfs_directories():
    log_message("Setting up HDFS directory structure...")
    directories = [
        HDFS_WEATHER_PATH,
        HDFS_TRAFFIC_PATH,
        HDFS_CLEANED_PATH
    ]
    for directory in directories:
        create_hdfs_directory(directory)
    log_message("HDFS directory structure created")
def setup_local_temp():
    if not os.path.exists(LOCAL_TEMP_DIR):
        os.makedirs(LOCAL_TEMP_DIR)
        log_message(f"Created local temp directory: {LOCAL_TEMP_DIR}")
def download_from_minio(bucket, object_name, local_path):
    log_message(f"Downloading {object_name} from MinIO {bucket} bucket...")
    try:
        client = get_client()
        client.fget_object(bucket, object_name, local_path)
        file_size = os.path.getsize(local_path)
        log_message(f"Downloaded: {object_name} ({file_size:,} bytes)")
        return True
    except Exception as e:
        log_message(f"Download failed: {e}")
        return False
def transfer_file_to_hdfs(local_path, hdfs_path, file_type):
    log_message(f"Transferring {file_type} to HDFS...")
    local_size = os.path.getsize(local_path)
    log_message(f"Local file size: {local_size:,} bytes")
    success = upload_to_hdfs(local_path, hdfs_path)
    if success:
        hdfs_size = get_hdfs_file_size(hdfs_path)
        log_message(f"HDFS file size: {hdfs_size:,} bytes")
        if hdfs_size == local_size:
            log_message(f"Transfer verified: {file_type}")
            return True
        else:
            log_message(f"Size mismatch for {file_type}")
            return False
    return False
def transfer_weather_data():
    log_message("\n" + "="*60)
    log_message("Transferring Weather Data")
    log_message("="*60)
    object_name = WEATHER_CLEANED_FILE
    local_path = os.path.join(LOCAL_TEMP_DIR, object_name)
    hdfs_path = f"{HDFS_WEATHER_PATH}/{object_name}"
    if not download_from_minio(MINIO_BUCKET_SILVER, object_name, local_path):
        return False
    if not transfer_file_to_hdfs(local_path, hdfs_path, "weather data"):
        return False
    hdfs_cleaned_path = f"{HDFS_CLEANED_PATH}/{object_name}"
    if not upload_to_hdfs(local_path, hdfs_cleaned_path):
        return False
    log_message("Weather data transfer complete")
    return True
def transfer_traffic_data():
    log_message("\n" + "="*60)
    log_message("Transferring Traffic Data")
    log_message("="*60)
    object_name = TRAFFIC_CLEANED_FILE
    local_path = os.path.join(LOCAL_TEMP_DIR, object_name)
    hdfs_path = f"{HDFS_TRAFFIC_PATH}/{object_name}"
    if not download_from_minio(MINIO_BUCKET_SILVER, object_name, local_path):
        return False
    if not transfer_file_to_hdfs(local_path, hdfs_path, "traffic data"):
        return False
    hdfs_cleaned_path = f"{HDFS_CLEANED_PATH}/{object_name}"
    if not upload_to_hdfs(local_path, hdfs_cleaned_path):
        return False
    log_message("Traffic data transfer complete")
    return True
def verify_hdfs_data():
    log_message("\n" + "="*60)
    log_message("Verifying HDFS Data")
    log_message("="*60)
    directories = {
        "Weather": HDFS_WEATHER_PATH,
        "Traffic": HDFS_TRAFFIC_PATH,
        "Cleaned": HDFS_CLEANED_PATH
    }
    all_verified = True
    for name, path in directories.items():
        log_message(f"\n{name} directory ({path}):")
        files = list_hdfs_directory(path)
        if files:
            for file_path in files:
                file_name = file_path.split('/')[-1]
                file_size = get_hdfs_file_size(file_path)
                log_message(f"  - {file_name}: {file_size:,} bytes")
        else:
            log_message(f"  No files found")
            all_verified = False
    return all_verified
def cleanup_temp():
    log_message("\nCleaning up temporary files...")
    try:
        if os.path.exists(LOCAL_TEMP_DIR):
            for file in os.listdir(LOCAL_TEMP_DIR):
                os.remove(os.path.join(LOCAL_TEMP_DIR, file))
            os.rmdir(LOCAL_TEMP_DIR)
            log_message("Temporary files cleaned up")
    except Exception as e:
        log_message(f"Cleanup warning: {e}")
def save_transfer_log():
    log_file = TRANSFER_LOG_FILE
    with open(log_file, 'w') as f:
        f.write("HDFS Data Transfer Log\n")
        f.write("="*60 + "\n\n")
        for entry in TRANSFER_LOG:
            f.write(entry + "\n")
    log_message(f"\nTransfer log saved to: {log_file}")
def main():
    start_time = time.time()
    log_message("="*60)
    log_message("HDFS Integration - Data Transfer Pipeline")
    log_message("="*60)
    log_message("\nChecking HDFS status...")
    if not check_hdfs_running():
        log_message("ERROR: HDFS is not running!")
        log_message("Please start HDFS using: /home/hamza/hadoop/sbin/start-dfs.sh")
        return
    log_message("HDFS is running")
    setup_hdfs_directories()
    setup_local_temp()
    weather_success = transfer_weather_data()
    traffic_success = transfer_traffic_data()
    if weather_success and traffic_success:
        verify_hdfs_data()
    cleanup_temp()
    save_transfer_log()
    elapsed_time = time.time() - start_time
    log_message("\n" + "="*60)
    log_message(f"Transfer completed in {elapsed_time:.2f} seconds")
    log_message("="*60)
    if weather_success and traffic_success:
        log_message("\nSUCCESS: All data transferred to HDFS")
        log_message("\nData locations:")
        log_message(f"  Weather: {HDFS_WEATHER_PATH}/weather_cleaned.parquet")
        log_message(f"  Traffic: {HDFS_TRAFFIC_PATH}/traffic_cleaned.parquet")
        log_message(f"  Cleaned: {HDFS_CLEANED_PATH}/")
    else:
        log_message("\nERROR: Some transfers failed. Check log for details.")
if __name__ == "__main__":
    main()
