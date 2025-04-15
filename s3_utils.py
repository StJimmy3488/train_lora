import boto3
from botocore.config import Config
import os
import logging
import shutil
from botocore.exceptions import NoCredentialsError

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


def upload_directory_to_s3(local_dir, bucket_name, s3_prefix):
    """Uploads a directory to S3 while maintaining folder structure"""
    logger.info("Uploading directory '%s' to S3 bucket '%s' with prefix '%s'.",
                local_dir, bucket_name, s3_prefix)
    try:
        s3 = get_s3_client()
        for root, _, files in os.walk(local_dir):
            for file in files:
                local_path = os.path.join(root, file)
                relative_path = os.path.relpath(local_path, local_dir)
                s3_key = f"{s3_prefix}/{relative_path}"
                logger.debug("Uploading file '%s' to '%s'.", local_path, s3_key)
                s3.upload_file(local_path, bucket_name, s3_key)
        return True
    except NoCredentialsError:
        logger.error("AWS credentials not available.")
        return False
    except Exception as e:
        logger.exception("Error uploading to S3: %s", e)
        return False
    finally:
        # Clean up local directory regardless of upload success
        logger.debug("Removing local directory '%s'.", local_dir)
        shutil.rmtree(local_dir, ignore_errors=True)
