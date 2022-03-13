# Django Imports
from django.contrib import admin
from django.urls import path, include
from django.conf import settings
from django.conf.urls.static import static

from recommender import views as rviews

urlpatterns = [
    path('admin/', admin.site.urls),
    path("api/", include("recommender.api.urls")),
    # path('populate_titles/', rviews.MostViewedTitlesView.as_view(), name="populate_titles")
]

if settings.DEBUG:
    urlpatterns += static(settings.STATIC_URL, document_root=settings.STATIC_ROOT)
    urlpatterns += static(settings.MEDIA_URL, document_root=settings.MEDIA_ROOT)