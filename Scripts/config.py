import os
HADOOP_HOME = "/home/hamza/hadoop"
HADOOP_BIN = os.path.join(HADOOP_HOME, "bin", "hadoop")
HDFS_BIN = os.path.join(HADOOP_HOME, "bin", "hdfs")
HDFS_BASE_PATH = "/user/bigdata"
HDFS_BRONZE_PATH = f"{HDFS_BASE_PATH}/bronze"
HDFS_BRONZE_WEATHER_PATH = f"{HDFS_BRONZE_PATH}/weather"
HDFS_BRONZE_TRAFFIC_PATH = f"{HDFS_BRONZE_PATH}/traffic"
HDFS_SILVER_PATH = f"{HDFS_BASE_PATH}/silver"
HDFS_SILVER_WEATHER_PATH = f"{HDFS_SILVER_PATH}/weather"
HDFS_SILVER_TRAFFIC_PATH = f"{HDFS_SILVER_PATH}/traffic"
HDFS_GOLD_PATH = f"{HDFS_BASE_PATH}/gold"
HDFS_WEATHER_PATH = HDFS_SILVER_WEATHER_PATH
HDFS_TRAFFIC_PATH = HDFS_SILVER_TRAFFIC_PATH
HDFS_CLEANED_PATH = HDFS_SILVER_PATH
HDFS_BACKUP_BASE = "/user/bigdata_backup"
MINIO_ENDPOINT = "localhost:9002"
MINIO_CONSOLE_URL = "http://localhost:9003"
MINIO_ACCESS_KEY = "bdprojectfcds4"
MINIO_SECRET_KEY = "bdprojectfcds4"
MINIO_SECURE = False
MINIO_BUCKET_BRONZE = "bronze"
MINIO_BUCKET_SILVER = "silver"
MINIO_BUCKET_GOLD = "gold"
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(PROJECT_ROOT, "Data")
TEMP_DIR = os.path.join(PROJECT_ROOT, "temp_hdfs")
VALIDATION_TEMP_DIR = os.path.join(PROJECT_ROOT, "temp_validation")
DOCUMENTATION_DIR = os.path.join(PROJECT_ROOT, "Documentation")
WEATHER_RAW_FILE = "weather_raw.csv"
TRAFFIC_RAW_FILE = "traffic_raw.csv"
WEATHER_CLEANED_FILE = "weather_cleaned.parquet"
TRAFFIC_CLEANED_FILE = "traffic_cleaned.parquet"
WEATHER_RAW_PATH = os.path.join(DATA_DIR, WEATHER_RAW_FILE)
TRAFFIC_RAW_PATH = os.path.join(DATA_DIR, TRAFFIC_RAW_FILE)
WEATHER_CLEANED_PATH = os.path.join(DATA_DIR, WEATHER_CLEANED_FILE)
TRAFFIC_CLEANED_PATH = os.path.join(DATA_DIR, TRAFFIC_CLEANED_FILE)
MAX_WORKERS = 4
BACKUP_KEEP_COUNT = 3
TRANSFER_LOG_FILE = os.path.join(DOCUMENTATION_DIR, "transfer_log.txt")
VALIDATION_REPORT_FILE = os.path.join(DOCUMENTATION_DIR, "validation_report.md")
def get_hadoop_command(command):
    return [HADOOP_BIN] + command.split()
def get_hdfs_command(command):
    return [HDFS_BIN] + command.split()
def ensure_dir(directory):
    if not os.path.exists(directory):
        os.makedirs(directory)
    return directory
def detect_environment():
    issues = []
    if not os.path.exists(HADOOP_HOME):
        issues.append(f"Hadoop not found at: {HADOOP_HOME}")
    if not os.path.exists(HADOOP_BIN):
        issues.append(f"Hadoop binary not found at: {HADOOP_BIN}")
    if not os.path.exists(HDFS_BIN):
        issues.append(f"HDFS binary not found at: {HDFS_BIN}")
    if not os.path.exists(DATA_DIR):
        issues.append(f"Data directory not found: {DATA_DIR}")
    return issues
def print_configuration():
    print("=" * 60)
    print("Current Configuration")
    print("=" * 60)
    print(f"\nHadoop:")
    print(f"  HADOOP_HOME: {HADOOP_HOME}")
    print(f"  HADOOP_BIN: {HADOOP_BIN}")
    print(f"  HDFS_BIN: {HDFS_BIN}")
    print(f"\nHDFS Paths:")
    print(f"  Base: {HDFS_BASE_PATH}")
    print(f"  Weather: {HDFS_WEATHER_PATH}")
    print(f"  Traffic: {HDFS_TRAFFIC_PATH}")
    print(f"  Cleaned: {HDFS_CLEANED_PATH}")
    print(f"\nMinIO:")
    print(f"  Endpoint: {MINIO_ENDPOINT}")
    print(f"  Console: {MINIO_CONSOLE_URL}")
    print(f"  Access Key: {MINIO_ACCESS_KEY}")
    print(f"\nLocal Paths:")
    print(f"  Project Root: {PROJECT_ROOT}")
    print(f"  Data Dir: {DATA_DIR}")
    print(f"  Temp Dir: {TEMP_DIR}")
    issues = detect_environment()
    if issues:
        print(f"\nConfiguration Issues:")
        for issue in issues:
            print(f"  ! {issue}")
    else:
        print(f"\nConfiguration: OK")
    print("=" * 60)
if __name__ == "__main__":
    print_configuration()
