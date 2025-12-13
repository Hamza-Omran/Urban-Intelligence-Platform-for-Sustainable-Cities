"""
Global Configuration File
All configuration variables that may change across different environments
Update this file when deploying to a different system
"""

import os

# ============================================
# Hadoop Configuration
# ============================================
HADOOP_HOME = "/home/hamza/hadoop"
HADOOP_BIN = os.path.join(HADOOP_HOME, "bin", "hadoop")
HDFS_BIN = os.path.join(HADOOP_HOME, "bin", "hdfs")

# ============================================
# HDFS Paths
# ============================================
HDFS_BASE_PATH = "/user/bigdata"
HDFS_WEATHER_PATH = f"{HDFS_BASE_PATH}/weather"
HDFS_TRAFFIC_PATH = f"{HDFS_BASE_PATH}/traffic"
HDFS_CLEANED_PATH = f"{HDFS_BASE_PATH}/cleaned"
HDFS_BACKUP_BASE = "/user/bigdata_backup"

# ============================================
# MinIO Configuration
# ============================================
MINIO_ENDPOINT = "localhost:9002"
MINIO_CONSOLE_URL = "http://localhost:9003"
MINIO_ACCESS_KEY = "bdprojectfcds4"
MINIO_SECRET_KEY = "bdprojectfcds4"
MINIO_SECURE = False

# MinIO Bucket Names
MINIO_BUCKET_BRONZE = "bronze"
MINIO_BUCKET_SILVER = "silver"
MINIO_BUCKET_GOLD = "gold"

# ============================================
# Local Paths
# ============================================
# Project root directory (auto-detected)
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Data directories
DATA_DIR = os.path.join(PROJECT_ROOT, "Data")
TEMP_DIR = os.path.join(PROJECT_ROOT, "temp_hdfs")
VALIDATION_TEMP_DIR = os.path.join(PROJECT_ROOT, "temp_validation")

# Documentation directory
DOCUMENTATION_DIR = os.path.join(PROJECT_ROOT, "Documentation")

# ============================================
# Data Files
# ============================================
# Raw data files
WEATHER_RAW_FILE = "weather_raw.csv"
TRAFFIC_RAW_FILE = "traffic_raw.csv"

# Cleaned data files
WEATHER_CLEANED_FILE = "weather_cleaned.parquet"
TRAFFIC_CLEANED_FILE = "traffic_cleaned.parquet"

# Local file paths
WEATHER_RAW_PATH = os.path.join(DATA_DIR, WEATHER_RAW_FILE)
TRAFFIC_RAW_PATH = os.path.join(DATA_DIR, TRAFFIC_RAW_FILE)
WEATHER_CLEANED_PATH = os.path.join(DATA_DIR, WEATHER_CLEANED_FILE)
TRAFFIC_CLEANED_PATH = os.path.join(DATA_DIR, TRAFFIC_CLEANED_FILE)

# ============================================
# Transfer Configuration
# ============================================
# Number of parallel workers for file transfers
MAX_WORKERS = 4

# Backup retention policy
BACKUP_KEEP_COUNT = 3

# ============================================
# Logging Configuration
# ============================================
TRANSFER_LOG_FILE = os.path.join(DOCUMENTATION_DIR, "transfer_log.txt")
VALIDATION_REPORT_FILE = os.path.join(DOCUMENTATION_DIR, "validation_report.md")

# ============================================
# Helper Functions
# ============================================
def get_hadoop_command(command):
    """Get full path to hadoop command"""
    return [HADOOP_BIN] + command.split()

def get_hdfs_command(command):
    """Get full path to hdfs command"""
    return [HDFS_BIN] + command.split()

def ensure_dir(directory):
    """Ensure directory exists"""
    if not os.path.exists(directory):
        os.makedirs(directory)
    return directory

# ============================================
# Environment Detection
# ============================================
def detect_environment():
    """Detect current environment and validate configuration"""
    issues = []
    
    # Check Hadoop installation
    if not os.path.exists(HADOOP_HOME):
        issues.append(f"Hadoop not found at: {HADOOP_HOME}")
    
    if not os.path.exists(HADOOP_BIN):
        issues.append(f"Hadoop binary not found at: {HADOOP_BIN}")
    
    if not os.path.exists(HDFS_BIN):
        issues.append(f"HDFS binary not found at: {HDFS_BIN}")
    
    # Check data directory
    if not os.path.exists(DATA_DIR):
        issues.append(f"Data directory not found: {DATA_DIR}")
    
    return issues

def print_configuration():
    """Print current configuration for debugging"""
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
    
    # Check for issues
    issues = detect_environment()
    if issues:
        print(f"\nConfiguration Issues:")
        for issue in issues:
            print(f"  ! {issue}")
    else:
        print(f"\nConfiguration: OK")
    
    print("=" * 60)

# ============================================
# Main
# ============================================
if __name__ == "__main__":
    print_configuration()
