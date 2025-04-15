import boto3
from botocore.config import Config
import os
import logging
import shutil
from botocore.exceptions import NoCredentialsError
import asyncio
logger = logging.getLogger(__name__)

def get_s3_client():
    """Create S3-compatible client for Cloudflare R2"""
    s3_endpoint = os.getenv("S3_ENDPOINT")
    aws_access_key_id = os.getenv("AWS_ACCESS_KEY_ID")
    aws_secret_access_key = os.getenv("AWS_SECRET_ACCESS_KEY")
    region_name = os.getenv("AWS_REGION")

    if not s3_endpoint or not aws_access_key_id or not aws_secret_access_key:
        raise RuntimeError("Missing S3 credentials. Check .env file.")

    config = Config(
        signature_version="s3v4", 
        retries={"max_attempts": 3, "mode": "standard"}
    )

    return boto3.client(
        "s3",
        endpoint_url=s3_endpoint,
        aws_access_key_id=aws_access_key_id,
        aws_secret_access_key=aws_secret_access_key,
        region_name=region_name,
        config=config
    )



async def upload_directory_to_s3(local_dir, bucket_name, s3_prefix):
    # Implement parallel uploads
    async def upload_file(file_path, s3_key):
        try:
            s3 = get_s3_client()
            s3.upload_file(file_path, bucket_name, s3_key)
            return True
        except Exception as e:
            logger.error(f"Failed to upload {file_path}: {e}")
            return False

    upload_tasks = []
    for root, _, files in os.walk(local_dir):
        for file in files:
            local_path = os.path.join(root, file)
            relative_path = os.path.relpath(local_path, local_dir)
            s3_key = f"{s3_prefix}/{relative_path}"
            upload_tasks.append(upload_file(local_path, s3_key))
    
    results = await asyncio.gather(*upload_tasks)
    return all(results)