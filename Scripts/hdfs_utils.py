import subprocess
import os
from config import HADOOP_BIN, HDFS_BIN
def run_hdfs_command(command):
    try:
        result = subprocess.run(
            command,
            capture_output=True,
            text=True,
            check=False
        )
        return result.returncode == 0, result.stdout, result.stderr
    except Exception as e:
        return False, "", str(e)
def create_hdfs_directory(path):
    command = [HDFS_BIN, "dfs", "-mkdir", "-p", path]
    success, output, error = run_hdfs_command(command)
    if success:
        print(f"Created HDFS directory: {path}")
    else:
        print(f"Directory already exists or created: {path}")
    return True
def upload_to_hdfs(local_path, hdfs_path):
    if not os.path.exists(local_path):
        print(f"Local file not found: {local_path}")
        return False
    remove_command = [HDFS_BIN, "dfs", "-rm", "-f", hdfs_path]
    run_hdfs_command(remove_command)
    command = [HDFS_BIN, "dfs", "-put", local_path, hdfs_path]
    success, output, error = run_hdfs_command(command)
    if success:
        print(f"Uploaded: {local_path} -> {hdfs_path}")
        return True
    else:
        print(f"Upload failed: {error}")
        return False
def list_hdfs_directory(path):
    command = [HDFS_BIN, "dfs", "-ls", path]
    success, output, error = run_hdfs_command(command)
    if not success:
        return []
    files = []
    for line in output.strip().split('\n'):
        if line and not line.startswith('Found'):
            parts = line.split()
            if len(parts) >= 8:
                files.append(parts[-1])
    return files
def get_hdfs_file_size(hdfs_path):
    command = [HDFS_BIN, "dfs", "-du", "-s", hdfs_path]
    success, output, error = run_hdfs_command(command)
    if success and output:
        parts = output.strip().split()
        if parts:
            return int(parts[0])
    return -1
def download_from_hdfs(hdfs_path, local_path):
    if os.path.exists(local_path):
        os.remove(local_path)
    command = [HDFS_BIN, "dfs", "-get", hdfs_path, local_path]
    success, output, error = run_hdfs_command(command)
    if success:
        print(f"Downloaded: {hdfs_path} -> {local_path}")
        return True
    else:
        print(f"Download failed: {error}")
        return False
def check_hdfs_running():
    command = [HDFS_BIN, "dfs", "-ls", "/"]
    success, output, error = run_hdfs_command(command)
    return success
