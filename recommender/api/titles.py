import io
import json
import os
import time

import boto3
import numpy as np
import scipy.sparse as sparse
import sagemaker.amazon.common as smac

# Django Imports
from django.conf import settings

# Django Rest Imports
from rest_framework import authentication, status
from rest_framework.response import Response
from rest_framework.views import APIView

# Import Utils
from recommender.services import Boto3FileDownload
from recommender.utils import MapTitleTextJSONFiles, LTokenizer, _get_spanish_stop_words
from recommender._stop_words_sp import SPANISH_WORDS as spanish_words

# Note: This can be change to another service, for the purpose
#       of this particular project the service is not commited
from recommender.content_service import get_service

# Model Imports
from recommender.models import Title

# Scikit Import
from sklearn.feature_extraction.text import CountVectorizer

VOCAB_SIZE = 2000
SAGEMAKER_BUCKET = "sagemaker-erecommender"
PROFILE_NAME = "prod"
PREFIX = "recommender"


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
    def post(self, request):
        body = json.loads(request.body)
        book_limit = body.get("book_limit", 0)
        list_keys = body.get("list_keys", [])
        theme_filter = body.get("theme", None)
        start_time = time.time()
        print("Started Token Vectorization!")
        vectorizer = CountVectorizer(
            input="content",
            analyzer="word",
            stop_words=spanish_words,
            tokenizer=LTokenizer(),
            max_features=VOCAB_SIZE,
            max_df=0.95,
            min_df=2
        )
        print("Started Tokenization")
        titles = self._get_titles_text(book_limit, list_keys, theme_filter)
        print(f"Titles length -> {len(titles)}")
        print("Transform!")
        vectors = vectorizer.fit_transform(titles)
        vocab_list = vectorizer.get_feature_names_out()
        print('vocab size:', len(vocab_list))

        # random shuffle
        index = np.arange(vectors.shape[0])
        new_index = np.random.permutation(index)

        # Need to store these permutations:
        vectors = vectors[new_index]
        enlapse_time = time.time() - start_time
        print('Done. Time elapsed: {:.2f}s'.format(enlapse_time))

        vectors = sparse.csr_matrix(vectors, dtype=np.float32)
        print(type(vectors), vectors.dtype)

        # Convert data into training and validation data
        n_train = int(0.8 * vectors.shape[0])

        # split train and test
        train_vectors = vectors[:n_train, :]
        val_vectors = vectors[n_train:, :]

        print(train_vectors.shape,val_vectors.shape)


        #define paths
        bucket = SAGEMAKER_BUCKET
        prefix = PREFIX

        train_prefix = os.path.join(prefix, 'train')
        val_prefix = os.path.join(prefix, 'val')
        output_prefix = os.path.join(prefix, 'output')

        s3_train_data = os.path.join('s3://', bucket, train_prefix)
        s3_val_data = os.path.join('s3://', bucket, val_prefix)
        output_path = os.path.join('s3://', bucket, output_prefix)
        print('Training set location', s3_train_data)
        print('Validation set location', s3_val_data)
        print('Trained model will be saved at', output_path)

        # Split the training and validation vectors
        self._split_convert_upload(
            train_vectors, bucket_name=bucket, prefix=train_prefix, fname_template='train_part{}.pbr', n_parts=8)
        self._split_convert_upload(
            val_vectors, bucket_name=bucket, prefix=val_prefix, fname_template='val_part{}.pbr', n_parts=1)

        return Response({"status": "OK"}, status=status.HTTP_200_OK)

    def _get_titles_text(self, book_limit, list_keys, theme_filter) -> list:
        """
        Function to create a list of text of the books mapped in the service
        :param book_limit int:
        :param list_keys  list:
        :param theme_filter str:
        :return list:
        """
        compiled_text = []
        if len(list_keys) > 0:
            titles = Title.objects.filter(identifier__in=list_keys)
        elif theme_filter:
            titles = Title.objects.filter(theme=theme_filter)
        else:
            titles = Title.objects.all().exclude(complete_text=u'').order_by("-pk")
        if book_limit > 0:
            titles = titles[:book_limit]
            print(f"Amount of titles -> {titles.count()}")
        for title in titles:
            if not title.complete_text == "":
                compiled_text.append(title.complete_text)
        return compiled_text

    def _split_convert_upload(self, sparray, bucket_name, prefix, fname_template='data_part{}.pbr', n_parts=2):
        chunk_size = sparray.shape[0]// n_parts
        prod = boto3.Session(profile_name=PROFILE_NAME, region_name="us-west-1")
        s3 = prod.resource('s3')
        bucket = s3.Bucket(bucket_name)
        for i in range(n_parts):
            # Calculate start and end indices
            start = i*chunk_size
            end = (i+1)*chunk_size
            if i+1 == n_parts:
                end = sparray.shape[0]

            # Convert to record protobuf
            buf = io.BytesIO()
            smac.write_spmatrix_to_sparse_tensor(array=sparray[start:end], file=buf, labels=None)
            buf.seek(0)

            # Upload to s3 location specified by bucket and prefix
            fname = os.path.join(prefix, fname_template.format(i))
            bucket.Object(fname).upload_fileobj(buf)
            print('Uploaded data to s3://{}'.format(os.path.join(bucket_name, fname)))
