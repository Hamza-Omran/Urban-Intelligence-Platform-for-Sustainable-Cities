"""
Upload Raw Data to Bronze Bucket
Uploads raw CSV files to MinIO bronze layer
"""

from minio_utils import get_client, upload_file

BUCKET = "bronze"

FILES_TO_UPLOAD = [
    ("./Data/weather_raw.csv", "weather_raw.csv"),
    ("./Data/traffic_raw.csv", "traffic_raw.csv"),
]


def main():
    client = get_client()
    
    for local_path, object_name in FILES_TO_UPLOAD:
        upload_file(client, BUCKET, local_path, object_name)


if __name__ == "__main__":
    main()
