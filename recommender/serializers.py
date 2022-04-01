# Serializers Import
from rest_framework import serializers

# Model Imports
from recommender.models import Recommendation, Title, Workflow


class TitleSerializer(serializers.ModelSerializer):
    class Meta:
        model = Title
        fields = "__all__"


class RecommendationSerializer(serializers.ModelSerializer):
    title = TitleSerializer()
    recommendation = TitleSerializer()

    class Meta:
        model = Recommendation
        fields = "__all__"


class WorkflowSerializer(serializers.ModelSerializer):
    class Meta:
        model = Workflow
        fields = "__all__"