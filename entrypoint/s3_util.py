import logging
import os
from typing import Optional
import uuid
from urllib.parse import urlparse

import boto3
from botocore.exceptions import ClientError

from entrypoint.models import S3Config


logger = logging.getLogger(__name__)


def get_s3_client(config: S3Config):
    s3_client = boto3.client(
        "s3", aws_access_key_id=config.aws_access_key_id, aws_secret_access_key=config.aws_secret_access_key
    )
    return s3_client


def upload_fileobj(s3_client, file, bucket, object_name):
    try:
        response = s3_client.upload_fileobj(file, bucket, object_name)
        logging.info(response)
    except ClientError as e:
        logging.error(e)
        return False
    return True


def upload_file_and_get_presigned_url(s3_client, config: S3Config, file_path: str, s3_object_prefix: Optional[str] = None):
    try:
        # Extract bucket and object name from config and file path
        bucket = config.bucket
        # Use the filename as object name
        filename = os.path.basename(file_path)
        
        # Construct the object_name with the provided s3_object_prefix if available,
        # otherwise use the default prefix_path from config.
        if s3_object_prefix:
            object_name = f"{s3_object_prefix.rstrip('/')}/{filename}"
        else:
            object_name = f"{config.prefix_path.rstrip('/')}/{filename}"
        
        # Open and upload the file
        with open(file_path, 'rb') as file:
            if not upload_fileobj(s3_client, file, bucket, object_name):
                logger.info(f"File {object_name} failed upload to {bucket}/{object_name}")
                return None
        
        logger.info(f"File {object_name} uploaded to {bucket}/{object_name}")
        
        # Delete the local file after successful upload
        os.remove(file_path)
        logger.info(f"Local file {file_path} deleted after upload")
        
        response_url = create_presigned_url(s3_client, bucket, object_name)
        if response_url is not None:
            logger.info(f"Presigned URL: {response_url}")
            return response_url
        else:
            logger.info("Presigned URL failed")
            return None
    except Exception as e:
        logger.error(f"Error uploading file {file_path}: {e}")
        return None


# Generate a presigned URL for the S3 object
def create_presigned_url(s3_client, bucket_name, object_name, expiration=3600):
    try:
        response_url = s3_client.generate_presigned_url(
            "get_object", Params={"Bucket": bucket_name, "Key": object_name}, ExpiresIn=expiration
        )
    except ClientError as e:
        logging.error(e)
        return None
    # The response contains the presigned URL
    return response_url


def download_s3_url_to_temp(s3_client, s3_url: str):
    """
    Download a file from an S3 URL to a local /temp/{uuid}/ directory and return the path.

    Args:
        s3_client: Authenticated S3 client
        s3_url (str): The S3 URL of the file to download. Supports both s3:// and https:// formats.

    Returns:
        str: Local file path where the file was downloaded, or None if failed.
    """
    try:
        # Parse the S3 URL to extract bucket name and object key
        parsed_url = urlparse(s3_url)
        
        # Handle both s3:// and https:// URL formats
        if parsed_url.scheme == 's3':
            # s3://bucket-name/object-key format
            bucket_name = parsed_url.netloc
            object_name = parsed_url.path.lstrip('/')
        elif parsed_url.scheme == 'https':
            # https://bucket-name.s3.region.amazonaws.com/object-key format
            # Extract bucket name from netloc (before .s3)
            bucket_name = parsed_url.netloc.split('.')[0]
            # Object key is the path without leading slash
            object_name = parsed_url.path.lstrip('/')
        else:
            logger.error(f"Invalid S3 URL scheme: {s3_url}. Expected 's3://' or 'https://'.")
            return None

        if not bucket_name or not object_name:
            logger.error(f"Could not extract bucket name or object name from S3 URL: {s3_url}")
            return None

        # Generate a unique UUID for the temp directory
        temp_uuid = str(uuid.uuid4())
        temp_dir = os.path.join("/tmp", temp_uuid) # Using /tmp for temporary files

        # Create the temp directory if it doesn't exist
        os.makedirs(temp_dir, exist_ok=True)

        # Get the filename from the object name
        filename = os.path.basename(object_name)
        local_file_path = os.path.join(temp_dir, filename)

        # Download the file from S3
        s3_client.download_file(bucket_name, object_name, local_file_path)

        logger.info(f"File downloaded from {s3_url} to {local_file_path}")
        return local_file_path

    except ClientError as e:
        logging.error(f"Error downloading file from S3: {e}")
        return None
    except Exception as e:
        logging.error(f"Unexpected error downloading file {s3_url}: {e}")
        return None

def download_s3_folder_to_temp(s3_client, s3_folder_url: str):
    """
    Download all files from an S3 folder URL to a local temporary directory.

    Args:
        s3_client: Authenticated S3 client.
        s3_folder_url (str): The S3 URL of the folder to download.

    Returns:
        str: Local path to the temporary directory where files were downloaded, or None if failed.
    """
    try:
        parsed_url = urlparse(s3_folder_url)
        
        if parsed_url.scheme == 's3':
            bucket_name = parsed_url.netloc
            prefix = parsed_url.path.lstrip('/')
        elif parsed_url.scheme == 'https':
            bucket_name = parsed_url.netloc.split('.')[0]
            prefix = parsed_url.path.lstrip('/')
        else:
            logger.error(f"Invalid S3 folder URL scheme: {s3_folder_url}. Expected 's3://' or 'https://'.")
            return None

        if not bucket_name or not prefix:
            logger.error(f"Could not extract bucket name or prefix from S3 folder URL: {s3_folder_url}")
            return None

        # Ensure prefix ends with a slash for listing objects within the folder
        if not prefix.endswith('/'):
            prefix += '/'

        temp_uuid = str(uuid.uuid4())
        local_folder_path = os.path.join("/tmp", temp_uuid)
        os.makedirs(local_folder_path, exist_ok=True)

        paginator = s3_client.get_paginator('list_objects_v2')
        pages = paginator.paginate(Bucket=bucket_name, Prefix=prefix)

        for page in pages:
            if "Contents" in page:
                for obj in page["Contents"]:
                    object_key = obj["Key"]
                    # Skip if it's a directory marker
                    if object_key.endswith('/'):
                        continue

                    # Construct local file path, preserving subdirectories
                    relative_path = os.path.relpath(object_key, prefix)
                    local_file_path = os.path.join(local_folder_path, relative_path)
                    
                    # Create parent directories if they don't exist
                    os.makedirs(os.path.dirname(local_file_path), exist_ok=True)

                    logger.info(f"Downloading s3://{bucket_name}/{object_key} to {local_file_path}")
                    s3_client.download_file(bucket_name, object_key, local_file_path)
        
        logger.info(f"Folder downloaded from {s3_folder_url} to {local_folder_path}")
        return local_folder_path

    except ClientError as e:
        logging.error(f"Error downloading folder from S3: {e}")
        return None
    except Exception as e:
        logging.error(f"Unexpected error downloading folder {s3_folder_url}: {e}")
        return None
