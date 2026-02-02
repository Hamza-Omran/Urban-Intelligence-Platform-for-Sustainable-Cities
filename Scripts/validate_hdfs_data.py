import os
import pandas as pd
from datetime import datetime
from hdfs_utils import (
    download_from_hdfs,
    get_hdfs_file_size,
    list_hdfs_directory
)
from config import (
    HDFS_BASE_PATH,
    VALIDATION_TEMP_DIR,
    VALIDATION_REPORT_FILE,
    ensure_dir
)
LOCAL_VALIDATION_DIR = VALIDATION_TEMP_DIR
VALIDATION_REPORT = []
def log_validation(message):
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    entry = f"[{timestamp}] {message}"
    print(entry)
    VALIDATION_REPORT.append(entry)
def validate_hdfs_structure():
    log_validation("Validating HDFS directory structure...")
    required_paths = [
        "/user/bigdata/weather",
        "/user/bigdata/traffic",
        "/user/bigdata/cleaned"
    ]
    all_valid = True
    for path in required_paths:
        files = list_hdfs_directory(path)
        if files is not None:
            log_validation(f"  Directory exists: {path}")
        else:
            log_validation(f"  MISSING directory: {path}")
            all_valid = False
    return all_valid
def validate_file_integrity(hdfs_path, file_description):
    log_validation(f"\nValidating {file_description}...")
    file_size = get_hdfs_file_size(hdfs_path)
    if file_size <= 0:
        log_validation(f"  ERROR: File not found or empty")
        return False
    log_validation(f"  File size: {file_size:,} bytes")
    if not os.path.exists(LOCAL_VALIDATION_DIR):
        os.makedirs(LOCAL_VALIDATION_DIR)
    local_file = os.path.join(LOCAL_VALIDATION_DIR, hdfs_path.split('/')[-1])
    if download_from_hdfs(hdfs_path, local_file):
        try:
            df = pd.read_parquet(local_file)
            row_count = len(df)
            col_count = len(df.columns)
            log_validation(f"  Parquet file valid")
            log_validation(f"  Rows: {row_count:,}")
            log_validation(f"  Columns: {col_count}")
            log_validation(f"  Columns: {', '.join(df.columns.tolist())}")
            os.remove(local_file)
            return True
        except Exception as e:
            log_validation(f"  ERROR: Cannot read parquet file: {e}")
            return False
    else:
        log_validation(f"  ERROR: Cannot download file")
        return False
def validate_all_files():
    log_validation("\n" + "="*60)
    log_validation("File Integrity Validation")
    log_validation("="*60)
    files_to_validate = [
        ("/user/bigdata/weather/weather_cleaned.parquet", "Weather data (weather dir)"),
        ("/user/bigdata/traffic/traffic_cleaned.parquet", "Traffic data (traffic dir)"),
        ("/user/bigdata/cleaned/weather_cleaned.parquet", "Weather data (cleaned dir)"),
        ("/user/bigdata/cleaned/traffic_cleaned.parquet", "Traffic data (cleaned dir)")
    ]
    all_valid = True
    for hdfs_path, description in files_to_validate:
        if not validate_file_integrity(hdfs_path, description):
            all_valid = False
    return all_valid
def generate_validation_report():
    log_validation("\n" + "="*60)
    log_validation("Validation Summary")
    log_validation("="*60)
    report_file = VALIDATION_REPORT_FILE
    with open(report_file, 'w') as f:
        f.write("# HDFS Data Validation Report\n\n")
        f.write(f"**Date:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        f.write("## Validation Log\n\n")
        f.write("```\n")
        for entry in VALIDATION_REPORT:
            f.write(entry + "\n")
        f.write("```\n")
    log_validation(f"\nValidation report saved to: {report_file}")
def cleanup():
    if os.path.exists(LOCAL_VALIDATION_DIR):
        try:
            os.rmdir(LOCAL_VALIDATION_DIR)
        except:
            pass
def main():
    log_validation("="*60)
    log_validation("HDFS Data Validation")
    log_validation("="*60)
    structure_valid = validate_hdfs_structure()
    if structure_valid:
        files_valid = validate_all_files()
    else:
        files_valid = False
        log_validation("\nSkipping file validation due to structure errors")
    generate_validation_report()
    cleanup()
    log_validation("\n" + "="*60)
    if structure_valid and files_valid:
        log_validation("VALIDATION PASSED: All data valid in HDFS")
    else:
        log_validation("VALIDATION FAILED: Check report for details")
    log_validation("="*60)
if __name__ == "__main__":
    main()
