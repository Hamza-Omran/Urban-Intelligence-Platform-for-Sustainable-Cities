import subprocess
from datetime import datetime
from hdfs_utils import run_hdfs_command, list_hdfs_directory
from config import HDFS_BASE_PATH, HDFS_BACKUP_BASE, BACKUP_KEEP_COUNT
def ensure_backup_directory():
    files = list_hdfs_directory(HDFS_BACKUP_BASE)
    if files is None:
        print(f"Backup directory doesn't exist, creating: {HDFS_BACKUP_BASE}")
        command = [
            "/home/hamza/hadoop/bin/hdfs",
            "dfs", "-mkdir", "-p",
            HDFS_BACKUP_BASE
        ]
        success, output, error = run_hdfs_command(command)
        if success:
            print(f"Backup directory created: {HDFS_BACKUP_BASE}")
            return True
        else:
            print(f"Failed to create backup directory: {error}")
            return False
    return True
def create_backup(backup_name=None):
    if not ensure_backup_directory():
        return False
    if backup_name is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        backup_name = f"backup_{timestamp}"
    backup_path = f"{HDFS_BACKUP_BASE}/{backup_name}"
    print(f"Creating backup: {backup_path}")
    command = [
        "/home/hamza/hadoop/bin/hdfs",
        "dfs", "-cp", "-p",
        HDFS_BASE_PATH,
        backup_path
    ]
    success, output, error = run_hdfs_command(command)
    if success:
        print(f"Backup created successfully: {backup_path}")
        return True
    else:
        print(f"Backup failed: {error}")
        return False
def list_backups():
    files = list_hdfs_directory(HDFS_BACKUP_BASE)
    if files:
        print(f"\nAvailable backups in {HDFS_BACKUP_BASE}:")
        for file_path in files:
            backup_name = file_path.split('/')[-1]
            print(f"  - {backup_name}")
        return files
    else:
        print("No backups found")
        return []
def restore_backup(backup_name):
    backup_path = f"{HDFS_BACKUP_BASE}/{backup_name}"
    restore_path = f"{HDFS_BASE_PATH}_restored_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    print(f"Restoring backup: {backup_name}")
    print(f"Restore location: {restore_path}")
    command = [
        "/home/hamza/hadoop/bin/hdfs",
        "dfs", "-cp", "-p",
        backup_path,
        restore_path
    ]
    success, output, error = run_hdfs_command(command)
    if success:
        print(f"Backup restored successfully to: {restore_path}")
        return True
    else:
        print(f"Restore failed: {error}")
        return False
def delete_old_backups(keep_count=BACKUP_KEEP_COUNT):
    backups = list_hdfs_directory(HDFS_BACKUP_BASE)
    if not backups or len(backups) <= keep_count:
        print(f"No old backups to delete (total: {len(backups) if backups else 0})")
        return 0
    backups.sort()
    backups_to_delete = backups[:-keep_count]
    deleted_count = 0
    for backup_path in backups_to_delete:
        backup_name = backup_path.split('/')[-1]
        print(f"Deleting old backup: {backup_name}")
        command = [
            "/home/hamza/hadoop/bin/hdfs",
            "dfs", "-rm", "-r",
            backup_path
        ]
        success, output, error = run_hdfs_command(command)
        if success:
            deleted_count += 1
    print(f"Deleted {deleted_count} old backups")
    return deleted_count
def verify_backup(backup_name):
    backup_path = f"{HDFS_BACKUP_BASE}/{backup_name}"
    expected_files = [
        "bronze/weather/weather_raw.csv",
        "bronze/traffic/traffic_raw.csv",
        "silver/weather/weather_cleaned.parquet",
        "silver/traffic/traffic_cleaned.parquet"
    ]
    print(f"\nVerifying backup: {backup_name}")
    all_found = True
    for expected_file in expected_files:
        full_path = f"{backup_path}/{expected_file}"
        files = list_hdfs_directory(full_path)
        if files:
            print(f"   Found: {expected_file}")
        else:
            print(f"   MISSING: {expected_file}")
            all_found = False
    if all_found:
        print(f"\nBackup verified: {backup_name}")
    else:
        print(f"\nBackup incomplete: {backup_name}")
    return all_found
def main():
    print("="*60)
    print("HDFS Backup Utility")
    print("="*60)
    print("\n1. Creating new backup...")
    backup_created = create_backup()
    if backup_created:
        print("\n2. Listing all backups...")
        backups = list_backups()
        if backups:
            latest_backup = backups[-1].split('/')[-1]
            print(f"\n3. Verifying latest backup: {latest_backup}")
            verify_backup(latest_backup)
    print("\n" + "="*60)
    print("Backup operations complete")
    print("="*60)
if __name__ == "__main__":
    main()
