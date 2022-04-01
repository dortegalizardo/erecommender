# Serializers Import
from rest_framework import serializers

# Model Imports
from recommender.models import Workflow


class WorkflowSerializer(serializers.ModelSerializer):
    class Meta:
        model = Workflow
        fields = "__all__"