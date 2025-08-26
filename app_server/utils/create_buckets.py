from minio_client import MinioClient


def create_buckets():
    buckets = ["accounts", "temporary-data", "user-registration", "face-activity-logs", "face-alarm-logs"]

    for bucket_name in buckets:
        try:
            status, message = MinioClient.create_bucket(bucket_name)
            if status:
                print(f"[SUCCESS] Bucket '{bucket_name}' created successfully.")
            else:
                print(f"[WARNING] Bucket '{bucket_name}' already exists or could not be created: {message}")
        except Exception as e:
            print(f"[ERROR] Error creating bucket '{bucket_name}': {str(e)}")


if __name__ == "__main__":
    create_buckets()
