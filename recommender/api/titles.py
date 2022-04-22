import io
import json
import os
import time
import joblib

import boto3
import numpy as np
import scipy.sparse as sparse
import sagemaker.amazon.common as smac

# Django Imports
from django.conf import settings
from django.core.files import File

# Django Rest Imports
from rest_framework import authentication, status
from rest_framework.response import Response
from rest_framework.views import APIView

# Import Utils
from recommender.services import Boto3FileDownload
from recommender.utils import MapTitleTextJSONFiles, LTokenizer, S3SessionMakerMixin
from recommender._stop_words_sp import SPANISH_WORDS as spanish_words

# Note: This can be change to another service, for the purpose
#       of this particular project the service is not commited
from recommender.content_service import get_service

# Model Imports
from recommender.models import Title, Workflow

# Serializers Import
from recommender.serializers import WorkflowSerializer

# Scikit Import
from sklearn.feature_extraction.text import CountVectorizer

# SageMaker Estimator
import sagemaker
from sagemaker.amazon.amazon_estimator import get_image_uri
from sagemaker.predictor import csv_serializer, json_deserializer
from sagemaker.session import s3_input

VOCAB_SIZE = 4000
SAGEMAKER_BUCKET = "sagemaker-erecommender"
PROFILE_NAME = "prod"
PREFIX = "recommender"
NUM_TOPICS=150
NUM_NEIGHBORS=10


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
        test_title = Title.objects.filter(identifier=title.get("sync_key"), training_book=False)
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
            queryset = Title.objects.filter(complete_text=u'')

        # Process all files
        map_service = MapTitleTextJSONFiles(settings.BOOK_PATH)

        for title in queryset:
            folder = f"{file_path}/{title.identifier}"
            merged_text, number_pages = map_service.process_json_file(folder=folder)
            title.complete_text = merged_text
            title.number_pages = number_pages
            title.save()
            print(f"Complete text saved for -> {title.name}")

        return Response({"status": "OK"}, status=status.HTTP_200_OK)


class PrepareTrainData(APIView, S3SessionMakerMixin):

    def post(self, request):
        # Get data from request
        body = json.loads(request.body)
        book_limit = body.get("book_limit", 0)
        list_keys = body.get("list_keys", [])
        theme_filter = body.get("theme", None)
        start_time = time.time()
        processed_documents = -452
        # Create Worflow
        workflow = Workflow()
        workflow.save()
        try:
            os.mkdir(f'{settings.BOOK_PATH}/{workflow.pk}/')
        except FileExistsError:
            print("File already exists!")

        print("Started Token Vectorization!")
        vectorizer = CountVectorizer(
            input="content",
            analyzer="word",
            stop_words=spanish_words,
            tokenizer=LTokenizer(processed_documents),
            max_features=VOCAB_SIZE,
            max_df=0.95,
            min_df=2
        )
        titles_train, titles_id = self._get_title_queryset(book_limit, list_keys, theme_filter)
        titles = self._get_titles_text(titles_train)

        workflow.booklist = {
            "training_ids": titles_id
        }

        print("Transform!")
        print("Started Tokenization...")
        vectors = vectorizer.fit_transform(titles)
        vocab_list = vectorizer.get_feature_names_out()

        # Map vectors to training titles
        self._map_vectors_titles(vectors=vectors, queryset=titles_train)

        # save numpy array
        np.savetxt(f'{settings.BOOK_PATH}/{workflow.pk}/vocab_list.csv', vocab_list, fmt='%s')
        self._assign_file_worklfow(
            file_path=f'{settings.BOOK_PATH}/{workflow.pk}/vocab_list.csv',
            file_name="vocab_list.csv",
            field=workflow.vocab_list
        )

        # random shuffle
        index = np.arange(vectors.shape[0])
        np.savetxt(f'{settings.BOOK_PATH}/{workflow.pk}/index.csv', index, fmt='%s')
        self._assign_file_worklfow(
            file_path=f'{settings.BOOK_PATH}/{workflow.pk}/index.csv',
            file_name="index.csv",
            field=workflow.index
        )

        new_index = np.random.permutation(index)
        np.savetxt(f'{settings.BOOK_PATH}/{workflow.pk}/new_index.csv', new_index, fmt='%s')
        self._assign_file_worklfow(
            file_path=f'{settings.BOOK_PATH}/{workflow.pk}/new_index.csv',
            file_name="new_index.csv",
            field=workflow.new_index
        )

        # Need to store these permutations:
        vectors = vectors[new_index]

        # Need to save the vector
        print("Saving vector in book path for later use.")
        joblib.dump(vectors, f"{settings.BOOK_PATH}/{workflow.pk}/vectors.joblib")
        workflow.training_vectors = f"{settings.BOOK_PATH}/{workflow.pk}/vectors.joblib"
        workflow.save()

        enlapse_time = time.time() - start_time
        workflow.processing_times = {"training_vectors_time": enlapse_time}
        workflow.save()

        print('Done. Time elapsed: {:.2f}s'.format(enlapse_time))

        vectors = sparse.csr_matrix(vectors, dtype=np.float32)
        print(type(vectors), vectors.dtype)

        # Convert data into training and validation data
        n_train = int(0.8 * vectors.shape[0])

        # split train and test
        train_vectors = vectors[:n_train, :]
        val_vectors = vectors[n_train:, :]

        print(train_vectors.shape,val_vectors.shape)

        # Define paths for training data
        bucket = SAGEMAKER_BUCKET
        prefix = f"{PREFIX}-WORKFLOW-{workflow.pk}"

        train_prefix = os.path.join(prefix, 'train')
        val_prefix = os.path.join(prefix, 'val')
        output_prefix = os.path.join(prefix, 'output')

        s3_train_data = os.path.join('s3://', bucket, train_prefix)
        s3_val_data = os.path.join('s3://', bucket, val_prefix)
        output_path = os.path.join('s3://', bucket, output_prefix)
        print('Training set location', s3_train_data)
        print('Validation set location', s3_val_data)
        print('Trained model will be saved at', output_path)

        workflow.s3_paths = {
            "training":
            {
                "s3_train_data": s3_train_data,
                "s3_val_data": s3_val_data,
                "output_path": output_path
            }
        }
        workflow.save()

        # Split the training and validation vectors
        self._split_convert_upload(
            train_vectors, bucket_name=bucket, prefix=train_prefix, fname_template='train_part{}.pbr', n_parts=8)
        self._split_convert_upload(
            val_vectors, bucket_name=bucket, prefix=val_prefix, fname_template='val_part{}.pbr', n_parts=1)

        serialize_data = WorkflowSerializer(workflow, many=False).data
        return Response(serialize_data, status=status.HTTP_200_OK)

    def _map_vectors_titles(self, vectors, queryset) -> None:
        """
        Function that assigns the right vector file to the title.
        :param vectors <ndarray>:
        :param queryset Title:
        :return None:
        """
        print("Starting vectors assignment to Titles...")
        item_counter = 0
        vectors = np.array(vectors.todense())
        for item in queryset:
            np.savetxt(
                f'{settings.BOOK_PATH}/{item.identifier}/vector_file.csv', vectors[item_counter], fmt='%s')
            local_file = open(f'{settings.BOOK_PATH}/{item.identifier}/vector_file.csv')
            parsed_file = File(local_file)
            item.vector_file.save('vector_file.csv', parsed_file)
            local_file.close()
            item_counter += 1
    
        print("Ended vectors assignment to Titles...")

    def _assign_file_worklfow(self, file_path: str, file_name:str, field: any) -> None:
        """
        Function that saves files for a particular workflow
        :param file_path str:
        :param file_name str:
        :param field Workflow field:
        :return None:
        """
        local_file = open(file_path)
        parsed_file = File(local_file)
        field.save(file_name, parsed_file)
        local_file.close()
        return None

    def _get_title_queryset(self, book_limit, list_keys, theme_filter):
        if len(list_keys) > 0:
            titles = Title.objects.filter(identifier__in=list_keys).exclude(complete_text=u'').order_by("pk")
        elif theme_filter:
            titles = Title.objects.filter(theme=theme_filter).exclude(complete_text=u'').order_by("pk")
        else:
            titles = Title.objects.filter(training_book=True).exclude(complete_text=u'').order_by("pk")
            print(f"The amount of titles is -> {titles.count()}")
        if book_limit > 0:
            titles = titles[:book_limit]
        titles_id = []
        for title in titles:
            titles_id.append(title.id)
        return titles, titles_id

    def _get_titles_text(self, queryset) -> list:
        """
        Function to create a list of text of the books mapped in the service
        :param queryset Title :
        :return list:
        """
        compiled_text = []
        for title in queryset:
            if not title.complete_text == "":
                compiled_text.append(title.complete_text)
        return compiled_text

    def _split_convert_upload(self, sparray, bucket_name, prefix, fname_template='data_part{}.pbr', n_parts=2):
        chunk_size = sparray.shape[0]// n_parts
        prod = self._get_boto_session()
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


class CreateNTMEstimator(APIView, S3SessionMakerMixin):
    def post(self, request, pk, *args, **kwargs):
        role, session = self._get_profile_role()
        container = get_image_uri(session.region_name, 'ntm')
        request_status = status.HTTP_200_OK

        # Get the Workflow that contains all the information
        workflow = Workflow.objects.filter(pk=pk)
        if not workflow.exists():
            return Response({
                "status": "ERROR",
                "message": "Workflow doesn't exist!"
            }, status=status.HTTP_404_NOT_FOUND)
        workflow = workflow.first()

        # Try to create TOPIC Estimator
        print("Starting Estimator Creation")
        try:
            session = sagemaker.Session(boto_session=session)
            ntm = sagemaker.estimator.Estimator(
                container,
                role,
                train_instance_count=2,
                train_instance_type="ml.c5.xlarge",
                output_path=workflow.s3_paths["training"]["output_path"],
                sagemaker_session=session
            )
            # set the hyperparameters for the topic model
            ntm.set_hyperparameters(
                num_topics=NUM_TOPICS,
                feature_dim=VOCAB_SIZE,
                mini_batch_size=128, 
                epochs=100,
                num_patience_epochs=5,
                tolerance=0.001
            )
            s3_train = s3_input(workflow.s3_paths["training"]["s3_train_data"], distribution="ShardedByS3Key")
            ntm.fit({'train': s3_train, 'test': workflow.s3_paths["training"]["s3_val_data"]})
            ntm_predictor = ntm.deploy(initial_instance_count=1, instance_type='ml.c5.xlarge')
            endpoint = ntm_predictor.__dict__["endpoint"]
        except Exception as e:
            response = {
                "status": "ERROR",
                "error": str(e.args[0])
            }
            request_status = status.HTTP_400_BAD_REQUEST
            ntm_predictor.delete_endpoint()
            return Response(response, status=request_status)

        # Response body in case everything ran smooth
        workflow.ntm_predictor_endpoint = endpoint
        workflow.save()
        serialize_data = WorkflowSerializer(workflow, many=False).data
        return Response(serialize_data, status=request_status)


class GetPredictorInformation(APIView, S3SessionMakerMixin):
    def post(self, request, pk, *args, **kwargs):

        # Get the Workflow that contains all the information
        workflow = Workflow.objects.filter(pk=pk)
        if not workflow.exists():
            return Response({
                "status": "ERROR",
                "message": "Workflow doesn't exist!"
            }, status=status.HTTP_404_NOT_FOUND)
        workflow = workflow.first()

        role, session = self._get_profile_role()
        vectors_path = workflow.training_vectors
        vectors = self._get_vector(vectors_path)
        predictor_endpoint = workflow.ntm_predictor_endpoint
        predictions = []
        sagemaker_session = sagemaker.Session(boto_session=session)
        try:
            ntm_predictor = sagemaker.predictor.RealTimePredictor(
                endpoint=predictor_endpoint,
                sagemaker_session=sagemaker_session
            )
            ntm_predictor.content_type = 'text/csv'
            ntm_predictor.serializer = csv_serializer
            ntm_predictor.deserializer = json_deserializer
        except Exception as e:
            response = {
                "status": "ERROR",
                "message": e
            }
            return Response(response, status=status.HTTP_400_BAD_REQUEST)
        
        for item in np.array(vectors.todense()):
            np.shape(item)
            results = ntm_predictor.predict(item)
            predictions.append(np.array([prediction['topic_weights'] for prediction in results['predictions']]))
        
        predictions = np.array([np.ndarray.flatten(x) for x in predictions])
        np.savetxt(f'{settings.BOOK_PATH}/{workflow.pk}/predictions.csv', predictions, fmt='%s')
        self._assign_file_worklfow(
            file_path=f'{settings.BOOK_PATH}/{workflow.pk}/predictions.csv',
            file_name="predictions.csv",
            field=workflow.topic_predictions
        )
        
        # self._assign_predictions(workflow, predictions) # TODO > Determine if this is needed
        workflow.save()
        serialize_data = WorkflowSerializer(workflow, many=False).data
        return Response(serialize_data, status=status.HTTP_200_OK)
    
    def _get_vector(self, vectors_path):
        """
        Function that gets the vectors file and loads it into memory
        :return vector -> scipy matrix:
        """
        vectors = joblib.load(vectors_path)
        vectors = sparse.csr_matrix(vectors, dtype=np.float32)
        return vectors

    def _assign_file_worklfow(self, file_path: str, file_name:str, field: any) -> None:
        """
        Function that saves files for a particular workflow
        :param file_path str:
        :param file_name str:
        :param field Workflow field:
        :return None:
        """
        local_file = open(file_path)
        parsed_file = File(local_file)
        field.save(file_name, parsed_file)
        local_file.close()
        return None
    
    def _assign_predictions(self, workflow: any, predictions: any) -> None:
        """
        Function that assigns an index prediction to the book list
        """
        book_list = workflow.booklist
        prediction_mapping = {}
        prediction_count = 0
        for book in book_list:
            prediction_mapping[str(book)] = predictions(prediction_count)
            prediction_count += 1
        workflow.booklist["predictions"] = prediction_mapping
        workflow.save()
        return None


class CreateKNNEstimator(APIView, S3SessionMakerMixin):

    def post(self, request, pk, *args, **kwargs):
        # Get the Workflow that contains all the information
        workflow = Workflow.objects.filter(pk=pk)
        if not workflow.exists():
            return Response({
                "status": "ERROR",
                "message": "Workflow doesn't exist!"
            }, status=status.HTTP_404_NOT_FOUND)
        workflow = workflow.first()

        role, session = self._get_profile_role()
        index = np.loadtxt(workflow.index.file,  dtype='int')
        new_index = np.loadtxt(workflow.new_index.file,  dtype='int')
        predictions = np.loadtxt(workflow.topic_predictions.file,  dtype='float')
        
        # Starting configuration process
        labels = new_index 
        labeldict = dict(zip(new_index,index))
        print('train_features shape = ', predictions.shape)
        print('train_labels shape = ', labels.shape)
        
        buf = io.BytesIO()
        smac.write_numpy_to_dense_tensor(buf, predictions, labels)
        buf.seek(0)

        bucket_name = SAGEMAKER_BUCKET
        prefix = f"{PREFIX}-WORKFLOW-{workflow.pk}"
        key = 'knn/train'

        # Uploading training data to knn/train
        prod = self._get_boto_session()
        s3 = prod.resource('s3')
        bucket = s3.Bucket(bucket_name)

        fname = os.path.join(prefix, key)
        bucket.Object(fname).upload_fileobj(buf)
        s3_train_data = 's3://{}/{}/{}'.format(bucket_name, prefix, key)
        print('uploaded training data location: {}'.format(s3_train_data))

        # Setting an output path
        output_path = 's3://' + bucket_name + '/' + prefix + '/knn/output'
        workflow.s3_paths["classifier_paths"] = {
            "output_path": output_path,
            "s3_train_data": s3_train_data
        }
        workflow.save()
        hyperparams = {
            'feature_dim': predictions.shape[1],
            'k': NUM_NEIGHBORS,
            'sample_size': predictions.shape[0],
            'predictor_type': 'classifier' ,
            'index_metric':'COSINE'
        }
        endpoint_name = ""
        try:
            session = sagemaker.Session(boto_session=session)
            knn = sagemaker.estimator.Estimator(get_image_uri(boto3.Session().region_name, "knn"),
                role,
                train_instance_count=1,
                train_instance_type='ml.c5.xlarge',
                output_path=output_path,
                sagemaker_session=session)
            knn.set_hyperparameters(**hyperparams)
    
            # train a model. fit_input contains the locations of the train and test data
            fit_input = {'train': s3_train_data}
            knn.fit(fit_input)

            instance_type = 'ml.c5.xlarge'
            endpoint_name = 'knn-ml-c5-xlarge-%s'% (str(time.time()).replace('.','-'))
        
            print('setting up the endpoint...')
            knn.deploy(
                initial_instance_count=1,
                instance_type=instance_type,
                endpoint_name=endpoint_name,
                accept="application/jsonlines; verbose=true"
            )

            print("KNN Deployed successfully!")
        except Exception as e:
            response = {
                "status": "ERROR",
                "message": e
            }
            return Response(response, status=status.HTTP_400_BAD_REQUEST)
        
        workflow.knn_predictor_endpoint = endpoint_name
        workflow.save()
        serialize_data = WorkflowSerializer(workflow, many=False).data
        return Response(serialize_data, status=status.HTTP_200_OK)


class CreateTestVectors(APIView):
    def post(self, request, pk, *args, **kwargs):
        # TODO > SIMPLY do this better.

        processed_documents = -452
        vectorizer = CountVectorizer(
            input="content",
            analyzer="word",
            stop_words=spanish_words,
            tokenizer=LTokenizer(processed_documents),
            max_features=VOCAB_SIZE,
            max_df=0.95,
            min_df=2
        )
        # Getting all the books and text needed.
        titles = Title.objects.filter(training_book=False).exclude(complete_text=u'').order_by("pk")
        print(f"Number of titles to vectorize -> {titles.count()}")
        tiltes_list = []
        for item in titles:
            tiltes_list.append(item.complete_text)
        
        
        vectors = vectorizer.fit_transform(tiltes_list)
        joblib.dump(vectors, f"{settings.BOOK_PATH}/{pk}/test_vectors.joblib")
        print(type(vectors))
        vocab_list = vectorizer.get_feature_names_out()
        return Response({}, status=status.HTTP_200_OK)
