import os
import boto3
import botocore

class Boto3FileDownload:
    def __init__(self, region_name, bucket_name, profile_name):
        self.region_name = region_name
        self.bucket_name = bucket_name
        self.profile_name = profile_name

    def download_file(self, filekey: str) -> None:
        base_path = '/books/'
        try:
            prod = boto3.Session(profile_name=self.profile_name, region_name=self.region_name)
        except botocore.exceptions.ProfileNotFound:
            print("The profile was not found. Check your environment variables or ~/.aws/credentials.")
        
        s3 = prod.resource('s3')
        bucket = s3.Bucket(self.bucket_name)

        # Create local new folder
        path = os.path.join(base_path, filekey)
        os.mkdir(path)
        print(f"New folder created -> {path}")
        print(f"Downloading file summary -> {filekey}")

        # Download book to destination path
        bucket.download_file(f"{filekey}/summary.json", path)

        # Testing if a file was download by requesting the size of the file.
        try:
            filesize = os.path.getsize(path)
        except OSError:
            print(f"The service was unable to download a summary for {filekey}")
        
        if filesize == 0:
            print(f"Maybe there is no summary for file > {filesize}")