import datetime
import mimetypes
import os
import ssl
from io import BytesIO
from pathlib import Path

import urllib3
from dotenv import load_dotenv
from minio import Minio, error
from minio.commonconfig import CopySource
from urllib3.util.timeout import Timeout

CURRENT_FILE = Path(__file__).resolve()
ENV_PATH = CURRENT_FILE.parents[2]
load_dotenv(ENV_PATH / ".env")


TIMEOUT_CONFIG = Timeout(
    connect=int(os.getenv("CONNECT_TIMEOUT")),
    read=int(os.getenv("READ_TIMEOUT")),
    total=int(os.getenv("TOTAL_TIMEOUT")),
)


POOL_KWARGS = {
    "timeout": TIMEOUT_CONFIG,
    "retries": urllib3.Retry(
        total=int(os.getenv("MAX_RETRIES")),
        backoff_factor=float(os.getenv("BACKOFF_FACTOR")),
        status_forcelist=[500, 502, 503, 504],
    ),
    "maxsize": int(os.getenv("POOL_MAXSIZE")),
    "block": bool(os.getenv("POOL_BLOCK")),
}


MINIO_ENDPOINT = os.getenv("MINIO_ENDPOINT")
MINIO_ACCESS_KEY = os.getenv("MINIO_ACCESS_KEY")
MINIO_SECRET_KEY = os.getenv("MINIO_SECRET_KEY")

ENABLE_SSL = os.getenv("ENABLE_SSL", "False").lower() == "true"
CA_PATH = os.getenv("CA_PATH", None)

if ENABLE_SSL and (not CA_PATH or not os.path.exists(CA_PATH)):
    raise ValueError("SSL is enabled but CA_PATH is not set or the file does not exist.")


class MinioClient:
    _client = None

    @classmethod
    def get_client(cls):
        """Get MinIO client instance with timeout configuration."""
        try:
            if not cls._client:
                if ENABLE_SSL:
                    context = ssl.create_default_context(cafile=CA_PATH)
                    context.check_hostname = False
                    POOL_KWARGS["ssl_context"] = context
                    http_client = urllib3.PoolManager(**POOL_KWARGS)
                    cls._client = Minio(
                        endpoint=MINIO_ENDPOINT,
                        access_key=MINIO_ACCESS_KEY,
                        secret_key=MINIO_SECRET_KEY,
                        secure=True,
                        http_client=http_client,
                    )
                else:
                    print(": ", POOL_KWARGS)
                    http_client = urllib3.PoolManager(**POOL_KWARGS)
                    cls._client = Minio(
                        endpoint=MINIO_ENDPOINT,
                        access_key=MINIO_ACCESS_KEY,
                        secret_key=MINIO_SECRET_KEY,
                        secure=False,
                        http_client=http_client,
                    )
        except error.S3Error as e:
            return {"status": False, "error": str(e)}

        return cls._client

    @classmethod
    def create_bucket(cls, bucket_name: str) -> tuple[bool, str]:
        """Create a bucket."""
        try:
            client = cls.get_client()
            if not client.bucket_exists(bucket_name):
                client.make_bucket(bucket_name)
                return True, f"Bucket '{bucket_name}' created successfully."
            else:
                return False, f"Bucket '{bucket_name}' already exists."
        except error.S3Error as e:
            return False, {"status": False, "error": str(e)}

    @classmethod
    def upload_object(
        cls, bucket_name: str, absolute_path_or_binary: str | bytes, s3_object_key: str, is_binary: bool = False
    ) -> tuple[bool, dict]:
        """
        If `is_content` is True, pass the binary content of the file.
        """
        try:
            client = cls.get_client()
            if is_binary:
                try:
                    mime_type = mimetypes.guess_type(s3_object_key)[0]
                    if mime_type is None:
                        mime_type = "application/octet-stream"
                    file_size = len(absolute_path_or_binary)
                    file_data_io = BytesIO(absolute_path_or_binary)
                    client.put_object(
                        bucket_name,
                        s3_object_key,
                        file_data_io,
                        length=file_size,
                        content_type=mime_type,
                    )
                except Exception as e:
                    return False, {"status": False, "error": str(e)}

            if not is_binary and os.path.exists(absolute_path_or_binary):
                mime_type, _ = mimetypes.guess_type(absolute_path_or_binary)
                if mime_type is None:
                    mime_type = "application/octet-stream"
                try:
                    with open(absolute_path_or_binary, "rb") as file_data:
                        file_size = os.path.getsize(absolute_path_or_binary)
                        client.put_object(
                            bucket_name,
                            s3_object_key,
                            file_data,
                            length=file_size,
                            content_type=mime_type,
                        )
                except FileNotFoundError:
                    return False, {"status": False, "error": f"File '{absolute_path_or_binary}' not found."}

            return True, {"status": True, "object_name": s3_object_key, "bucket_name": bucket_name}
        except error.S3Error as e:
            return False, {"status": False, "error": str(e)}

    @classmethod
    def get_object_url(cls, bucket_name: str, object_name: str, expires_in_sec: int = 300) -> tuple[bool, dict]:
        """Get a pre-signed URL for a file."""
        try:
            client = cls.get_client()
            client.stat_object(bucket_name, object_name)
            url = client.presigned_get_object(
                bucket_name,
                object_name,
                expires=datetime.timedelta(seconds=expires_in_sec),
                response_headers={"response-cache-control": f"max-age={expires_in_sec}, public"},
            )
            return True, {"status": True, "url": url}
        except error.S3Error as e:
            return False, {"status": False, "error": str(e)}

    @classmethod
    def get_multiple_objects_url(
        cls, bucket_name: str, object_names: list[str], expires_in_sec: int = 300
    ) -> tuple[bool, dict]:
        """Get pre-signed URLs for multiple files."""
        try:
            client = cls.get_client()
            urls = {}
            for object_name in object_names:
                url = client.presigned_get_object(
                    bucket_name,
                    object_name,
                    expires=datetime.timedelta(seconds=expires_in_sec),
                    response_headers={"response-cache-control": f"max-age={expires_in_sec}, public"},
                )
                urls[object_name] = url
            return True, {"status": True, "urls": urls}
        except error.S3Error as e:
            return False, {"status": False, "error": str(e)}

    @classmethod
    def move_to_new_bucket(
        cls, source_bucket: str, destination_bucket: str, object_name: str, new_prefix: str = None
    ) -> tuple[bool, dict]:
        try:
            client = cls.get_client()
            if new_prefix:
                path_parts = object_name.split("/", 1)
                if len(path_parts) > 1:
                    new_object_name = f"{new_prefix}/{path_parts[1]}"
                else:
                    new_object_name = f"{new_prefix}/{object_name}"
            else:
                new_object_name = object_name
            source = CopySource(
                bucket_name=source_bucket,
                object_name=object_name,
            )
            client.copy_object(destination_bucket, new_object_name, source)
            client.remove_object(source_bucket, object_name)
            return True, {"status": True, "original_object_name": object_name, "new_object_name": new_object_name}
        except error.S3Error as e:
            return False, {"status": False, "error": str(e)}

    @classmethod
    def download_object(cls, bucket_name: str, object_name: str, download_path: str) -> tuple[bool, dict]:
        """Download a file from MinIO."""
        try:
            client = cls.get_client()
            response = client.get_object(bucket_name, object_name)
            with open(download_path, "wb") as file_data:
                for d in response.stream(64 * 1024):
                    file_data.write(d)
            response.close()
            response.release_conn()
            return True, {"status": True, "object_name": object_name, "download_path": download_path}
        except error.S3Error as e:
            return False, {"status": False, "error": str(e)}

    @classmethod
    def delete_object(cls, bucket_name: str, object_name: str) -> tuple[bool, dict]:
        """Delete a file from MinIO."""
        try:
            client = cls.get_client()
            client.remove_object(bucket_name, object_name)
            return True, {"status": True, "object_name": object_name}
        except error.S3Error as e:
            return False, {"status": False, "error": str(e)}

    @classmethod
    def delete_directory(cls, bucket_name: str, directory_name: str) -> tuple[bool, dict]:
        """Delete all objects in a directory."""
        try:
            client = cls.get_client()
            objects_to_delete = list(client.list_objects(bucket_name, prefix=directory_name, recursive=True))
            for obj in objects_to_delete:
                client.remove_object(bucket_name, obj.object_name)
            return True, {"status": True, "directory_name": directory_name}
        except error.S3Error as e:
            return False, {"status": False, "error": str(e)}

    @classmethod
    def cleanup_old_files(cls, bucket_name: str, max_age_seconds: int) -> dict:
        """
        Delete files older than max_age_seconds in the given bucket.
        Returns a dict with deleted file names.
        """
        client = cls.get_client()
        now = datetime.now(datetime.timezone.utc)
        deleted_files = []

        try:
            for obj in client.list_objects(bucket_name, recursive=True):
                file_age = (now - obj.last_modified).total_seconds()
                if file_age > max_age_seconds:
                    client.remove_object(bucket_name, obj.object_name)
                    deleted_files.append(obj.object_name)
                    print(f"Deleted old file: {obj.object_name}")
            return {"status": True, "deleted_files": deleted_files}
        except error.S3Error as e:
            print(f"Error cleaning up files: {e}")
            return {"status": False, "error": str(e)}
