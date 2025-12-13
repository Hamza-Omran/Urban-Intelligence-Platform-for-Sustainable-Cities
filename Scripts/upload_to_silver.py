"""
Upload Cleaned Data to Silver Bucket
Uploads cleaned parquet files to MinIO silver layer
"""

from minio_utils import get_client, upload_file

BUCKET = "silver"

FILES_TO_UPLOAD = [
    ("./Data/weather_cleaned.parquet", "weather_cleaned.parquet"),
    ("./Data/traffic_cleaned.parquet", "traffic_cleaned.parquet"),
]


def main():
    client = get_client()
    
    for local_path, object_name in FILES_TO_UPLOAD:
        upload_file(client, BUCKET, local_path, object_name)
    
    print("\nAll cleaned data uploaded to silver bucket!")


if __name__ == "__main__":
    main()
