from minio import Minio
from minio.error import S3Error
import os
from config import MINIO_ENDPOINT, MINIO_ACCESS_KEY, MINIO_SECRET_KEY, MINIO_SECURE
def get_client():
    return Minio(
        MINIO_ENDPOINT,
        access_key=MINIO_ACCESS_KEY,
        secret_key=MINIO_SECRET_KEY,
        secure=MINIO_SECURE
    )
def create_bucket(client, bucket_name):
    try:
        if client.bucket_exists(bucket_name):
            print(f"Bucket '{bucket_name}' already exists")
            return True
        else:
            client.make_bucket(bucket_name)
            print(f"Created bucket: '{bucket_name}'")
            return True
    except S3Error as e:
        print(f"Error with bucket '{bucket_name}': {e}")
        return False
def upload_file(client, bucket_name, local_path, object_name):
    try:
        if not os.path.exists(local_path):
            print(f"File not found: {local_path}")
            return False
        print(f"Uploading {local_path} -> {bucket_name}/{object_name}")
        client.fput_object(bucket_name, object_name, local_path)
        print(f"Uploaded: {object_name}")
        return True
    except S3Error as e:
        print(f"Upload failed for {object_name}: {e}")
        return False
def list_bucket_objects(client, bucket_name):
    try:
        objects = client.list_objects(bucket_name)
        object_list = []
        for obj in objects:
            object_list.append({
                'name': obj.object_name,
                'size': obj.size,
                'size_mb': obj.size / (1024 * 1024),
                'last_modified': obj.last_modified
            })
        return object_list
    except S3Error as e:
        print(f"Error listing bucket '{bucket_name}': {e}")
        return []
def list_all_buckets(client):
    try:
        return client.list_buckets()
    except S3Error as e:
        print(f"Error listing buckets: {e}")
        return []
