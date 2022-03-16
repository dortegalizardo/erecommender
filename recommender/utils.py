from typing import Any

import json
import os
import re
import random
import requests
import stanza
import numpy as np
import unicodedata

from pathlib import Path

# Other Imports
from recommender._stop_words_sp import SPANISH_WORDS


FORMAT = 'utf8'

TYPE_UPOS = ['NOUN', 'PRON', 'PROPN', 'ADV', 'ADJ']


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
    def __init__(self, stanza_path=None, lang='es'):
        if stanza_path is None:
            stanza_path = os.path.join(str(Path.home()), 'stanza_resources')
        try:
            stanza.download(lang, model_dir=stanza_path)
        except Exception as e:
            print(e)
            raise Exception(f"Language: '{lang}' is not supported by stanza. Try specifying another language")
        self.nlp = stanza.Pipeline('es')

    def __call__(self, doc):
        list_words = []
        testing_data = self.nlp(doc)
        for sentence in testing_data.sentences:
            for token in sentence.tokens:
                if (token.end_char - token.start_char) > 3 and re.match("[a-z].*", token.words[0].text) and token.words[0].upos in TYPE_UPOS:
                    list_words.append(token.words[0].to_dict()['lemma'])
        print("Document finished!")
        return list_words


class MapTitleTextJSONFiles:
    def __init__(self, file_path = "/books/"):
        self.file_path = file_path

    def get_list_files(self):
        return [os.path.join(self.file_path, f) for f in os.listdir(self.file_path)]

    def process_json_file(self, folder) -> str:
        summary = None
        merged_text = ''
        try:
            with open(f"{folder}/summary.json") as file:
                summary = json.load(file)
                random_keys = self._random_keys(summary)
                merged_text = self._string_merge(summary, random_keys)
                # parsed_text = self._parsed_text(merged_text) TODO > No sé que hacer con esto ya!
        except Exception as e:  # Errores ligados a abrir el archivo temporal.
            print(f"Something went wrong! {e}")
        print("Processing has finished!")
        return merged_text.replace("- ","")

    def _random_keys(self, summary:any) -> list:
        random_keys = []
        amount_keys = len(summary)
        random_keys = list(range(amount_keys)) 
        random.shuffle(random_keys)
        if amount_keys < 10:
            return random_keys
        return random_keys[0:10]

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