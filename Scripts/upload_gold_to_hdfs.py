import os
from datetime import datetime
from hdfs_utils import (
    create_hdfs_directory,
    upload_to_hdfs,
    list_hdfs_directory,
    get_hdfs_file_size
)
from config import (
    HDFS_GOLD_PATH,
    DATA_DIR
)
def log_message(message):
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"[{timestamp}] {message}")
def create_gold_structure():
    log_message("Creating Gold layer directory structure...")
    directories = [
        HDFS_GOLD_PATH,
        f"{HDFS_GOLD_PATH}/factor_analysis",
        f"{HDFS_GOLD_PATH}/simulation"
    ]
    for directory in directories:
        log_message(f"  Creating: {directory}")
        create_hdfs_directory(directory)
    log_message("Gold directory structure created!\n")
def upload_gold_files():
    log_message("=" * 60)
    log_message("GOLD LAYER - Uploading Aggregated Data")
    log_message("=" * 60)
    gold_local_path = os.path.join(DATA_DIR, "gold")
    if not os.path.exists(gold_local_path):
        log_message(f" Gold data directory not found: {gold_local_path}")
        return False
    gold_files = [f for f in os.listdir(gold_local_path) if os.path.isfile(os.path.join(gold_local_path, f))]
    if not gold_files:
        log_message("No gold files found to upload")
        return False
    log_message(f"\nFound {len(gold_files)} files to upload\n")
    factor_analysis_files = [f for f in gold_files if 'factor' in f.lower() or 'biplot' in f.lower() or 'scree' in f.lower()]
    simulation_files = [f for f in gold_files if 'simulation' in f.lower() or 'scenario' in f.lower() or 'probability' in f.lower() or 'risk' in f.lower() or 'congestion' in f.lower() or 'accident' in f.lower()]
    uploaded_count = 0
    if factor_analysis_files:
        log_message(" Factor Analysis Files:")
        for filename in factor_analysis_files:
            local_path = os.path.join(gold_local_path, filename)
            hdfs_path = f"{HDFS_GOLD_PATH}/factor_analysis/{filename}"
            if upload_to_hdfs(local_path, hdfs_path):
                size = get_hdfs_file_size(hdfs_path)
                log_message(f"   {filename}: {size:,} bytes")
                uploaded_count += 1
            else:
                log_message(f"   Failed: {filename}")
        log_message("")
    if simulation_files:
        log_message(" Simulation & Risk Analysis Files:")
        for filename in simulation_files:
            local_path = os.path.join(gold_local_path, filename)
            hdfs_path = f"{HDFS_GOLD_PATH}/simulation/{filename}"
            if upload_to_hdfs(local_path, hdfs_path):
                size = get_hdfs_file_size(hdfs_path)
                log_message(f"   {filename}: {size:,} bytes")
                uploaded_count += 1
            else:
                log_message(f"   Failed: {filename}")
        log_message("")
    other_files = [f for f in gold_files if f not in factor_analysis_files and f not in simulation_files]
    if other_files:
        log_message(" Other Gold Files:")
        for filename in other_files:
            local_path = os.path.join(gold_local_path, filename)
            hdfs_path = f"{HDFS_GOLD_PATH}/{filename}"
            if upload_to_hdfs(local_path, hdfs_path):
                size = get_hdfs_file_size(hdfs_path)
                log_message(f"   {filename}: {size:,} bytes")
                uploaded_count += 1
            else:
                log_message(f"   Failed: {filename}")
    log_message(f"\n Successfully uploaded {uploaded_count}/{len(gold_files)} files")
    return uploaded_count > 0
def verify_gold_layer():
    log_message("\n" + "=" * 60)
    log_message("VERIFICATION - Gold Layer")
    log_message("=" * 60)
    subdirs = ["factor_analysis", "simulation"]
    for subdir in subdirs:
        subdir_path = f"{HDFS_GOLD_PATH}/{subdir}"
        log_message(f"\n {subdir_path}:")
        files = list_hdfs_directory(subdir_path)
        if files:
            for file_path in files:
                file_name = file_path.split('/')[-1]
                file_size = get_hdfs_file_size(file_path)
                log_message(f"    - {file_name}: {file_size:,} bytes")
        else:
            log_message("    (empty)")
    log_message(f"\n {HDFS_GOLD_PATH} (root):")
    root_files = list_hdfs_directory(HDFS_GOLD_PATH)
    if root_files:
        for file_path in root_files:
            if not file_path.endswith("factor_analysis") and not file_path.endswith("simulation"):
                file_name = file_path.split('/')[-1]
                try:
                    file_size = get_hdfs_file_size(file_path)
                    log_message(f"    - {file_name}: {file_size:,} bytes")
                except:
                    pass
def main():
    log_message("=" * 60)
    log_message("HDFS Gold Layer Upload")
    log_message("=" * 60)
    log_message("Uploading aggregated and business-ready data\n")
    create_gold_structure()
    success = upload_gold_files()
    if success:
        verify_gold_layer()
    log_message("\n" + "=" * 60)
    log_message("Gold Layer Upload Complete!")
    log_message("=" * 60)
    log_message("\n Your Gold layer now contains:")
    log_message("  • Factor Analysis results")
    log_message("  • Monte Carlo Simulation outputs")
    log_message("  • Risk analysis and visualizations")
    log_message("  • Scenario analysis data")
if __name__ == "__main__":
    main()
