"""
HDFS Utilities Module
Common functions for interacting with local HDFS installation
"""

import subprocess
import os
from config import HADOOP_BIN, HDFS_BIN


def run_hdfs_command(command):
    """
    Execute an HDFS command
    
    Args:
        command (list): Command as list of strings
        
    Returns:
        tuple: (success, output, error)
    """
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
    """
    Create directory in HDFS
    
    Args:
        path (str): HDFS path to create
        
    Returns:
        bool: True if successful
    """
    command = [HDFS_BIN, "dfs", "-mkdir", "-p", path]
    success, output, error = run_hdfs_command(command)
    
    if success:
        print(f"Created HDFS directory: {path}")
    else:
        print(f"Directory already exists or created: {path}")
    
    return True


def upload_to_hdfs(local_path, hdfs_path):
    """
    Upload file to HDFS
    
    Args:
        local_path (str): Local file path
        hdfs_path (str): Destination HDFS path
        
    Returns:
        bool: True if successful
    """
    if not os.path.exists(local_path):
        print(f"Local file not found: {local_path}")
        return False
    
    # Remove existing file in HDFS if it exists
    remove_command = [HDFS_BIN, "dfs", "-rm", "-f", hdfs_path]
    run_hdfs_command(remove_command)
    
    # Upload file
    command = [HDFS_BIN, "dfs", "-put", local_path, hdfs_path]
    success, output, error = run_hdfs_command(command)
    
    if success:
        print(f"Uploaded: {local_path} -> {hdfs_path}")
        return True
    else:
        print(f"Upload failed: {error}")
        return False


def list_hdfs_directory(path):
    """
    List contents of HDFS directory
    
    Args:
        path (str): HDFS directory path
        
    Returns:
        list: List of file/directory names
    """
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
    """
    Get file size in HDFS
    
    Args:
        hdfs_path (str): HDFS file path
        
    Returns:
        int: File size in bytes, -1 if error
    """
    command = [HDFS_BIN, "dfs", "-du", "-s", hdfs_path]
    success, output, error = run_hdfs_command(command)
    
    if success and output:
        parts = output.strip().split()
        if parts:
            return int(parts[0])
    
    return -1


def download_from_hdfs(hdfs_path, local_path):
    """
    Download file from HDFS
    
    Args:
        hdfs_path (str): HDFS file path
        local_path (str): Local destination path
        
    Returns:
        bool: True if successful
    """
    # Remove local file if exists
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
    """
    Check if HDFS is running
    
    Returns:
        bool: True if HDFS is running
    """
    command = [HDFS_BIN, "dfs", "-ls", "/"]
    success, output, error = run_hdfs_command(command)
    return success
