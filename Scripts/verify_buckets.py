"""
Verify MinIO Bucket Contents
Lists all files in bronze, silver, and gold buckets
"""

from minio_utils import get_client, list_bucket_objects


def main():
    client = get_client()
    buckets = ['bronze', 'silver', 'gold']
    
    print("=" * 60)
    print("MinIO Bucket Contents Verification")
    print("=" * 60)
    print()
    
    for bucket_name in buckets:
        print(f"Bucket: {bucket_name}")
        print("-" * 60)
        
        objects = list_bucket_objects(client, bucket_name)
        
        if objects:
            for obj in objects:
                print(f"   {obj['name']}")
                print(f"      Size: {obj['size_mb']:.2f} MB ({obj['size']:,} bytes)")
                print(f"      Last Modified: {obj['last_modified']}")
                print()
        else:
            print(f"   Bucket is empty")
            print()
    
    print("=" * 60)


if __name__ == "__main__":
    main()
