from typing import Any

import json
import os
import re
import random
import requests
import stanza
import numpy as np
import boto3
import unicodedata

from pathlib import Path

# SageMaker
from sagemaker import get_execution_role

# Other Imports
from recommender._stop_words_sp import SPANISH_WORDS


FORMAT = 'utf8'

TYPE_UPOS = ['NOUN', 'PRON', 'ADV']
PROFILE_NAME = "prod"

NUMBER_PAGES = 40


def _get_spanish_stop_words(unaccented: bool=True) -> list:
    """
    function that returns a list of spanish stopwords.
    The function can return the list as unaccented or accented
    it all depends on the boolean parameter it receives
    :param unaccented:
    :return list:
    """
    if unaccented:
        list_words = []
        for word in SPANISH_WORDS:
            nkfd_form = unicodedata.normalize("NFKD", word)
            list_words.append(nkfd_form.encode("ASCII", "ignore").decode("ASCII"))
        return list_words
    return SPANISH_WORDS


class LTokenizer(object):
    def __init__(self, processed_documents, stanza_path=None, lang='es'):
        if stanza_path is None:
            stanza_path = os.path.join(str(Path.home()), 'stanza_resources')
        try:
            stanza.download(lang, model_dir=stanza_path)
        except Exception as e:
            print(e)
            raise Exception(f"Language: '{lang}' is not supported by stanza. Try specifying another language")
        self.nlp = stanza.Pipeline('es')
        self.processed_documents = processed_documents

    def __call__(self, doc):
        list_words = []
        testing_data = self.nlp(doc)
        for sentence in testing_data.sentences:
            for token in sentence.tokens:
                if (token.end_char - token.start_char) > 3 and re.match("[a-z].*", token.words[0].text) and token.words[0].upos in TYPE_UPOS:
                    list_words.append(token.words[0].to_dict()['lemma'])
        self.processed_documents += 1
        print(f"Document finished -> {self.processed_documents}")
        return list_words


class MapTitleTextJSONFiles:
    def __init__(self, file_path = "/books/"):
        self.file_path = file_path

    def get_list_files(self):
        return [os.path.join(self.file_path, f) for f in os.listdir(self.file_path)]

    def process_json_file(self, folder) -> str:
        summary = None
        merged_text = ''
        number_pages = 0
        try:
            with open(f"{folder}/summary.json") as file:
                summary = json.load(file)
                number_pages = len(summary)
                random_keys = self._random_keys(summary)
                merged_text = self._string_merge(summary, random_keys)
        except Exception as e:
            print(f"Something went wrong! {e}")
        print("Processing has finished!")
        return merged_text.replace("- ",""), number_pages

    def _random_keys(self, summary:any) -> list:
        random_keys = []
        amount_keys = len(summary)
        random_keys = list(range(amount_keys)) 
        random.shuffle(random_keys)
        if amount_keys < NUMBER_PAGES:
            return random_keys
        return random_keys[0:NUMBER_PAGES]

    def _string_merge(self, summary: any, random_keys: list) -> str:
        '''
        Functions that will string together all the pages of a summary
        :param: summary 
        :return: string
        '''
        complete_text = ""
        for key, value in summary.items():
            # Decode the text value of each item
            try:
                if int(key) in random_keys:
                    encoded_value = value.encode(FORMAT)
                    decoded_value = encoded_value.decode('utf-8', errors="replace").replace("\x00", "\uFFFD")
                    complete_text += decoded_value
            except Exception as e:
                # Fallback in case ther is nothing to decode
                if int(key) in random_keys:
                    complete_text += value
                continue
        return complete_text


class TokenAuth(requests.auth.AuthBase):
    """
    Token de autenticación para hacer peticiones a Content Center.
    """

    def __init__(self, auth_token):
        self._auth_token = auth_token

    def __call__(self, r):
        r.headers["Authorization"] = f"Token {self._auth_token}"
        return r


class S3SessionMakerMixin:
    """
    Mixin created to help handle the creation of boto Sessions and roles
    """

    def _get_boto_session(self):
        return boto3.Session(profile_name=PROFILE_NAME, region_name="us-west-1")

    def _get_profile_role(self):
        """
        Function to fetch the right role to interact with Sagemaker
        :return sagemaker role:
        :return boto3 session:
        """
        role = ""
        session = ""
        try:
            role = get_execution_role()
        except ValueError:
            session = boto3.Session(profile_name=PROFILE_NAME, region_name="us-west-1")
            iam = session.client('iam')
            role = iam.get_role(RoleName='SageMakerRole')['Role']['Arn']
        return role, session