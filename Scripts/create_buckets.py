"""
Create MinIO Buckets
Creates the three required buckets: bronze, silver, gold
"""

from minio_utils import get_client, create_bucket, list_all_buckets

BUCKETS = ["bronze", "silver", "gold"]


def main():
    print("Creating MinIO buckets...\n")
    
    client = get_client()
    
    for bucket_name in BUCKETS:
        create_bucket(client, bucket_name)
    
    print("\nAll buckets are ready!")
    
    # List all buckets to verify
    print("\nCurrent buckets:")
    buckets = list_all_buckets(client)
    for bucket in buckets:
        print(f"  - {bucket.name} (created: {bucket.creation_date})")


if __name__ == "__main__":
    main()
