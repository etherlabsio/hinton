import os
import logging
from boto3 import client

logger = logging.getLogger(__name__)
logging.getLogger("boto3").setLevel(logging.CRITICAL)
logging.getLogger("botocore").setLevel(logging.CRITICAL)
logging.getLogger("s3transfer").setLevel(logging.CRITICAL)
logging.getLogger("urllib3").setLevel(logging.CRITICAL)


class S3Manager(object):
    """
    Common class for performing operations on S3
    """

    def __init__(self, *args, **kwargs):
        self.bucket_name = kwargs.get("bucket_name")
        self.conn = client("s3")

    def upload_to_s3(self, object_name, file_name=None):
        """
        Upload given file to s3.
        Args:
            object_name:
            file_name:
        Returns:
        """
        s3_client = self.conn

        if file_name is None:
            file_name = object_name

        try:
            s3_client.upload_file(file_name, self.bucket_name, object_name)
        except Exception as e:
            logger.error("s3 upload failed", extra={"err": e})

    def upload_object(self, body, s3_key):
        """
        Upload object to s3 key
        Args:
            body:
            s3_key:
        Returns:
        """
        s3_client = self.conn
        try:
            s3_client.put_object(Body=body, Key=s3_key, Bucket=self.bucket_name)
            return True
        except Exception as e:
            logger.error("s3 upload failed", extra={"err": e})
            return False

    def download_file(self, file_name, download_dir=None):
        """
        Download a file given s3 prefix
        inside /tmp directory.
        Args:
            download_dir: Defaults to None. Specify if need to download the file to disk
            file_name: Points to the object name in S3 bucket
        Returns:
            file_obj (byte_string): Returns the download file object if `download_dir` is not specified.
            file_name (str): Download path if `download_dir` is specified
        """
        s3_client = self.conn
        file_name = file_name.split("tmp/")[-1]

        file_name_only = file_name.split("/")[-1]
        file_name_only_len = len(file_name_only)
        file_name_len = len(file_name)

        if download_dir is None:
            # Download the file as an object
            try:
                file_obj = s3_client.get_object(Bucket=self.bucket_name, Key=file_name)
                return file_obj
            except Exception as e:
                logger.error(
                    "Cannot download file", extra={"err": e, "fileName": file_name}
                )
                return
        else:
            file_dir = download_dir + file_name[0 : file_name_len - file_name_only_len]
            if not os.path.exists(file_dir):
                os.makedirs(file_dir)
            try:
                s3_client.download_file(
                    self.bucket_name, file_name, download_dir + file_name
                )
                return file_name
            except Exception as e:
                logger.error(
                    "Cannot download file", extra={"err": e, "fileName": file_name}
                )
                return

    def get_s3_results(self, dir_name):
        """
        Return all contents of a given dir in s3.
        Goes through the pagination to obtain all file names.
        Args:
            dir_name:
        Returns:
            bucket_object_list: List of objects in an S3 Bucket
        """
        dir_name = dir_name.split("tmp/")[-1]
        paginator = self.conn.get_paginator("list_objects")
        s3_results = paginator.paginate(
            Bucket=self.bucket_name,
            Prefix=dir_name,
            PaginationConfig={"PageSize": 1000},
        )
        bucket_object_list = []
        for page in s3_results:
            if "Contents" in page:
                for key in page["Contents"]:
                    s3_file_name = key["Key"].split("/")[-1]
                    bucket_object_list.append(s3_file_name)
        return bucket_object_list
