"""
This entire file is here because of latin and in order to be able to lemmatize and filter seeds before using them :)
"""

import os
import re
from typing import List

import cltk

from cltk.core.data_types import Pipeline, Process
from cltk.lemmatize.lat import LatinBackoffLemmatizer
from cltk.nlp import NLP
from cltk.tokenizers import LatinTokenizationProcess

cltk.data.fetch.FetchCorpus(language="lat").import_corpus("lat_models_cltk")
LEMMATIZER = LatinBackoffLemmatizer()


def remove_digits(text):
    return re.sub(r"\d", "", text)


def isRomanNumeral(word):
    return word and word[0] == "%"


# Turning it into a set, so that checking membership is O(1) time
ADDITIONAL_STOPWORDS = ["naturaris", "natural"]
STOPS = set(cltk.stops.lat.STOPS + ADDITIONAL_STOPWORDS)


def remove_stopwords(words: List[str]) -> List[str]:
    """
    Removes latin stopwords from a sequence of words.

    Parameters
    ----------
    words: list of str
        Words to filter.

    Returns
    ----------
    words: list of str
        All the words that are not stopwords in the sequence
    """
    return [word for word in words if word not in STOPS]


LEMMATIZATION_EXCEPTIONS = {
    "caritas",
}


def lemmatize(words):
    """
    Removes stopwords and lemmatizes a list of words.

    Parameters
    ----------
    words: list of str
        Words to process

    Returns
    ----------
    lemmata: list of str
        List of lemmata
    """

    # lemmata = remove_stopwords(words)
    lemmata = [
        token if token in LEMMATIZATION_EXCEPTIONS else lemma
        for token, lemma in LEMMATIZER.lemmatize(words)
    ]
    bad_tags = {"punc", "-que", "e", "-ne"}
    lemmata = [
        remove_digits(lemma)
        for lemma in lemmata
        if lemma and not isRomanNumeral(lemma) and lemma not in bad_tags
    ]
    lemmata = remove_stopwords(lemmata)
    return lemmata


def normalize(text):
    # Augustine specific tag removal
    text = re.sub(r"\[.*\]", "", text)
    text = re.sub(r"{\d", "", text)
    text = re.sub(r"}\d", "", text)
    text = re.sub(r"\(\(.*\)\)", "", text)
    # Exchanges question marks, semicolons and exclamation marks for dots
    text = re.sub(r"[\?!;]", ".", text)
    # removes commas
    text = re.sub(r",", "", text)
    # removes digits
    text = re.sub(r"\d", "", text)
    # replaces line endings with spaces
    text = re.sub(r"\n", " ", text)
    # removes unnecessary whitespace
    return text.strip()


TOKENIZER = NLP(
    language="lat",
    custom_pipeline=Pipeline(
        language="lat",
        description="Tokenizer Pipeline",
        processes=[LatinTokenizationProcess],
    ),
)


def tokenize(text):
    return TOKENIZER.analyze(text=text).tokens


def prepare_seeds(model, seeds):
    seeds = [seed.lower() for seed in seeds]
    return [seed for seed in lemmatize(seeds) if seed in model.wv.key_to_index]
