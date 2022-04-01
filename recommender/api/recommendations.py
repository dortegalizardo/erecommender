import json
import numpy as np

# Django Rest Imports
from rest_framework import status
from rest_framework.response import Response
from rest_framework.views import APIView

# Model Imports
from recommender.models import Title, Workflow, Recommendation

# Serializers Imports
from recommender.serializers import RecommendationSerializer

# Scikit Import
from sklearn.feature_extraction.text import CountVectorizer

# Utils Import
from recommender.utils import LTokenizer, S3SessionMakerMixin
from recommender._stop_words_sp import SPANISH_WORDS as spanish_words

# SageMaker Estimator
import sagemaker
from sagemaker.amazon.amazon_estimator import get_image_uri
from sagemaker.predictor import csv_serializer, json_deserializer
from sagemaker.session import s3_input

# TODO > Need to figure out if this is going to be saved some place o
# if this is going to be mapped in the settings file
VOCAB_SIZE = 2000
SAGEMAKER_BUCKET = "sagemaker-erecommender"
PROFILE_NAME = "prod"
PREFIX = "recommender"
NUM_TOPICS=50
NUM_NEIGHBORS=5

class GetRecommendationAPIView(APIView, S3SessionMakerMixin):
    
    def post(self, request, pk, *args, **kwargs):
        body = json.loads(request.body)
        role, session = self._get_profile_role()
        # First get the book you want to create recommendations
        try:
            current_title = Title.objects.get(pk=pk)
        except Title.DoesNotExist:
            response = {
                "status": "ERROR",
                "message": "Title not found!"
            }
            return Response(response, status=status.HTTP_404_NOT_FOUND)

        # Test if the book already has recommendations generated
        recommendations_set = Recommendation.objects.filter(title=current_title)
        if recommendations_set.exists():
            serialized_data = RecommendationSerializer(recommendations_set, many=True)
            return Response(serialized_data, status=status.HTTP_200_OK)

        # Get the workflow you are going to use to get the predictions label
        try:
            current_workflow = Workflow.objects.get(uuid_identifier=body["workflow"])
        except Workflow.DoesNotExist:
            response = {
                "status": "ERROR",
                "message": "Worfklow not found!"
            }
            return Response(response, status=status.HTTP_404_NOT_FOUND)

        if current_title.complete_text =='':
            response = {
                "status": "ERROR",
                "message": "The book doesnt have text to vectorize!"
            }
            return Response(response, status=status.HTTP_400_BAD_REQUEST)

        # Content vectorization 
        vectorizer = CountVectorizer(
            input="content",
            analyzer="word",
            stop_words=spanish_words,
            tokenizer=LTokenizer(),
            max_features=VOCAB_SIZE,
            max_df=0.95,
            min_df=2
        )
        vector = vectorizer.fit_transform([current_title.complete_text])
        test_vector = np.array(vector.todense())

        # Setting up the estimator
        predictor_endpoint = current_workflow.knn_predictor_endpoint
        try:
            session = sagemaker.Session(boto_session=session)
            knn_predictor = sagemaker.predictor.RealTimePredictor(
                endpoint=predictor_endpoint,
                sagemaker_session=session,
                accept="application/jsonlines; verbose=true"
            )
            knn_predictor.content_type = 'text/csv'
            knn_predictor.serializer = csv_serializer
            knn_predictor.deserializer = json_deserializer
        except Exception as e:
            response = {
                "status": "ERROR",
                "message": "Unable to get the KNN Predictor!"
            }
            return Response(response, status=status.HTTP_400_BAD_REQUEST)

        # Navigate through the list of vectors, there should be only one.
        test_result = {}
        for vec in test_vector:
            test_result = knn_predictor.predict(vec)

        # Generate all the recommendations for the requested book
        order=0
        book_list = current_workflow.booklist["training_ids"]
        for item in test_result["labels"]:
            recommendation = Recommendation(
                title=current_title,
                recommendation=self._get_title(book_list(int(item))),
                order=order
            )
            recommendation.save()
            order+=1
        recommendations_set = Recommendation.objects.filter(title=current_title).order_by("order")
        serialized_data = RecommendationSerializer(recommendations_set, many=True).data

        return Response(serialized_data, status=status.HTTP_200_OK)

    def _get_title(self, title_id:int) -> Title:
        return Title.object.get(pk=title_id)