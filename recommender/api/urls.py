from django.urls import path

from rest_framework.authtoken.views import obtain_auth_token
from recommender.api import titles as api_views


urlpatterns = [
    path('get-token/', obtain_auth_token, name='api_token_auth'),
    path("downloadtitles/", api_views.DownloadTitles.as_view(), name="get-titles-s3"),
    path("maptitleinfo/", api_views.MapTitleInformation.as_view(), name="map-title-info"),
    path("mapjsoninfo/", api_views.GetTextJSONFiles.as_view(), name="map-json-files"),
    path("prepare_data/", api_views.PrepareTrainData.as_view(), name="prepare-data")
]