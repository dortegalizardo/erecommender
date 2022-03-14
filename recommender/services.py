import os
import boto3
import botocore

from django.conf import settings

class Boto3FileDownload:
    def __init__(self, region_name, bucket_name, profile_name):
        self.region_name = region_name
        self.bucket_name = bucket_name
        self.profile_name = profile_name

    def download_file(self, filekey: str) -> None:
        """
        Function that provides help on downloading files from S3
        to the base path of BOOK_PATH
        """
        base_path = settings.BOOK_PATH
        try:
            prod = boto3.Session(profile_name=self.profile_name, region_name=self.region_name)
        except botocore.exceptions.ProfileNotFound:
            print("The profile was not found. Check your environment variables or ~/.aws/credentials.")
        
        s3 = prod.resource('s3')
        bucket = s3.Bucket(self.bucket_name)

        # Create local new folder
        path = os.path.join(base_path, filekey)

        # small fallback in case the folder already exists
        if os.path.exists(path):
            print("Path exists!")
            return None

        try:
            os.mkdir(path)
        except FileExistsError:
            print("File already exists!")

        print(f"New folder created -> {path}")
        print(f"Downloading file summary -> {filekey}")

        # Download book to destination path
        try:
            bucket.download_file(f"{filekey}/summary.json", str(path+"/summary.json"))
        except botocore.exceptions.ClientError:
            print(f"{filekey} -> Probably not found!")
            raise ValueError

        # Testing if a file was download by requesting the size of the file.
        try:
            filesize = os.path.getsize(path)
        except OSError:
            print(f"The service was unable to download a summary for {filekey}")
        
        if filesize == 0:
            print(f"Maybe there is no summary for file > {filesize}")
