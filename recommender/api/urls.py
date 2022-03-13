from django.urls import path

from rest_framework.authtoken.views import obtain_auth_token
from recommender.api import titles as api_views


urlpatterns = [
    path('get-token/', obtain_auth_token, name='api_token_auth'),
    path("downloadtitles/", api_views.DownloadTitles.as_view(), name="get-titles-s3"),
]