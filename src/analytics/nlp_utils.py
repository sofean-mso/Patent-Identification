# Copyright 2025 FIZ-Karlsruhe (Mustafa Sofean)

import requests
import json
from src.analytics.preprocess import *
import spacy
import unidecode
import os
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from dotenv import load_dotenv


load_dotenv()
ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
GENERAL_STOPWORD_PATH = os.path.join(ROOT_DIR, 'sources/stopwords.txt')
ALL_STOPWORD_PATH = os.path.join(ROOT_DIR, 'sources/stopwords-all.txt')

# Initializing the spaCy model instance
spacy_md = spacy.load('en_core_web_md', disable=['ner'])
spacy_md.max_length = 2000000
EMBEDDING_ENDPOINT = os.getenv('EMBEDDING_ENDPOINT')


PATENT_UNITS = ['k', 'h', 'v', 'wt', 'wt.', 'mhz', 'khz', 'ghz', 'hz', 'days', 'weeks', 'hours',
                    'minutes', 'seconds', 't', 'mpa', 'gpa', 'at.', 'mol.', 'at', 'm', 'n', 's-1', 'vol.',
                    'vol', 'ev', 'a', 'atm', 'bar', 'koe', 'oe', 'h.', 'mwcm−2', 'kev', 'mev', 'mev', 'day',
                    'week', 'hour', 'minute', 'month', 'months', 'year', 'cycles', 'years', 'fs', 'ns', 'ps',
                    'rpm', 'g', 'mg', 'macm−2', 'ma', 'mk', 'mt', 's-1', 'db', 'ag-1', 'mag-1', 'mag−1', 'mag',
                    'mah', 'mahg−1', 'm-2', 'mj', 'kj', 'm2g−1', 'thz', 'khz', 'kjmol−1', 'torr', 'gl-1', 'vcm−1',
                    'mvs−1', 'j', 'gj', 'mtorr', 'cm2', 'mbar', 'kbar', 'mmol', 'mol', 'moll−1', 'mω',
                    'ω', 'kω', 'mω', 'mgl−1', 'moldm−3', 'm2', 'm3', 'cm-1', 'cm', 'scm−1', 'acm−1', 'ev-1cm-2',
                    'cm-2', 'sccm', 'cm−2ev−1', 'cm-3ev-1', 'ka', 's−1', 'emu', 'l', 'cmhz1', 'gmol−1', 'kvcm-1',
                    'mpam1', 'cm2v-1s-1', 'acm−2', 'cm−2s−1', 'mv', 'ionscm−2', 'jcm−2', 'ncm−2', 'jcm−2', 'wcm−2',
                    'gwcm−2', 'acm-2k-2', 'gcm−3', 'cm3g-1', 'mgl−1', 'mgml−1', 'mgcm−2', 'mωcm', 'cm−2s−1', 'cm−2',
                    'ions', 'moll−1', 'nmol', 'psi', 'jkg-1k-1', 'km', 'wm−2', 'mass', 'mmhg', 'mmmin−1',
                    'gev', 'm−2', 'm-2s-1', 'kmin−1', 'gl−1', 'ng', 'hr', 'w', 'mn', 'kn', 'mrad', 'rad', 'arcsec',
                    'ag−1', 'dpa', 'cdm−2', 'cd', 'mcd', 'mhz', 'm−3', 'ppm', 'phr', 'ml', 'ml', 'mlmin−1', 'mwm−2',
                    'wm-1k-1', 'kwh', 'wkg−1', 'jm−3', 'm-3', 'gl−1', 'a−1', 'ks−1', 'mgdm−3', 'mms−1',
                    'ks', 'appm', 'ºc', 'hv', 'kda', 'da', 'kg', 'kgy', 'mgy', 'gy', 'mgy', 'gbps', 'μb', 'μl',
                    'μf', 'nf', 'pf', 'mf', 'a', 'å', 'a˚', 'μgl−1', 'mgl-1']

def remove_accent(txt):
    """Removes accents from a string.
    Args:
        txt: The input string.
    Returns:
        The de-accented string.
    """
    # There is a problem with angstrom sometimes, so ignoring length 1 strings.
    return unidecode.unidecode(txt) if len(txt) > 1 else txt


    return np.split()



def clean_corpus(corpus):
    """
    clean a collection of text documents
    :param corpus:
    :return:
    """
    cleaned_corpus = list()
    for doc in corpus:
        cleaned_corpus.append(clean_text(doc, spacy_md))

    return cleaned_corpus


def get_embeddings(text:str, model:str = 'PATENT_BERT'):
    """
        send a text and get related vector
        :param model:
        :param txt:
        :return:
        """
    sentence = [text]
    payload = {"sentences": sentence,
               "transformer_model": model,
               "normalize": True
               }
    resp = requests.post(EMBEDDING_ENDPOINT, data=json.dumps(payload))
    if resp.status_code != 200:
        print('Post /query/ {}'.format(resp.status_code))
    response_json = resp.json()
    embed_vector = response_json['embeddings'][0]
    print('An Embeddings vector has been extracted..')
    return embed_vector





