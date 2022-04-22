# Django Imports
from django.contrib import admin

# Model Imports
from recommender.models import Title, Recommendation, Workflow


@admin.register(Title)
class TitleAdmin(admin.ModelAdmin):
    list_display = ("name", "publisher", "theme", "identifier",)
    list_display_links = ("name", )
    list_filter = ("publisher", "theme",)
    search_fields = ("name", "publisher", "theme", "identifier")


@admin.register(Recommendation)
class RecommendationAdmin(admin.ModelAdmin):
    list_display = ("title", "recommendation", "distance",)
    list_display_links = ("title",)
    search_fields = ("title__name",)


@admin.register(Workflow)
class WorkflowAdmin(admin.ModelAdmin):
    list_display = ("uuid_identifier",)
    list_display_links = ("uuid_identifier",)