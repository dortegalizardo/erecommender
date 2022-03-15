import time
import json

# Django Imports
from django.conf import settings

# Django Rest Imports
from rest_framework import authentication, status
from rest_framework.response import Response
from rest_framework.views import APIView

# Import Utils
from recommender.services import Boto3FileDownload
from recommender.utils import MapTitleTextJSONFiles

# Note: This can be change to another service, for the purpose
#       of this particular project the service is not commited 
from recommender.content_service import get_service

# Model Imports
from recommender.models import Title


class DownloadTitles(APIView):
    authentication_classes = [authentication.TokenAuthentication]
    def post(self, request):
        body = json.loads(request.body)
        profile_name = body["profile_name"]
        bucket_name = body["bucket_name"]
        region_name = body["region_name"]
        list_keys = body["list_keys"]
        service = Boto3FileDownload(profile_name=profile_name, bucket_name=bucket_name, region_name=region_name)
        start_time = time.time()
        errors = []
        for key in list_keys:
            try:
                service.download_file(key)
            except ValueError:
                errors.append(key)

        end_time = time.time() - start_time
        response = {
            "status": "OK",
            "duration": end_time,
            "errors": len(errors)
        }
        return Response(response, status=status.HTTP_200_OK)


class MapTitleInformation(APIView):
    def post(self, request):
        body = json.loads(request.body)
        has_service = body["has_service"]
        list_keys = body.get("list_keys", None)
        if has_service:
            service = get_service()
            for key in list_keys:
                title = service.get_title(id=key)
                if title:
                    self._create_title_from_service(title)
        else:
            for item in body["titles"]:
                self._create_title(item)
        return Response({"status": "OK"}, status=status.HTTP_200_OK)

    def _create_title(self, item):
        raise NotImplementedError("Method not implemented yet")
    
    def _create_title_from_service(self, title: dict) -> None:
        test_title = Title.objects.filter(identifier=title.get("sync_key"))
        if not test_title.exists():
            if title.get("theme"):
                theme = title.get("theme")[0]["name"]
            else:
                theme = ""
            data = {
                "identifier": title.get("sync_key"),
                "publisher": title.get("publisher")["name"],
                "theme": theme,
                "name": title.get("title_name"),
            }
            new_title = Title(**data)
            new_title.save()


class GetTextJSONFiles(APIView):
    def post(self, request):
        body = json.loads(request.body)
        file_path = body.get("file_path", settings.BOOK_PATH)
        process_all = body.get("process_all", False)
        if process_all:
            queryset = Title.objects.all()
        else:
            queryset = Title.objects.filter(complete_text="")

        # Process all files
        map_service = MapTitleTextJSONFiles(settings.BOOK_PATH)
        
        for title in queryset:
            folder = f"{file_path}/{title.identifier}"
            merged_text = map_service.process_json_file(folder=folder)
            title.complete_text = merged_text
            title.save()
            print(f"Complete text saved for -> {title.name}")

        return Response({"status": "OK"}, status=status.HTTP_200_OK)


class PrepareTrainData(APIView):
    pass
