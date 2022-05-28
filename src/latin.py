"""
Module containing useful utilites for processing latin texts.
"""
import re
from typing import List

import cltk
from cltk.core.data_types import Pipeline
from cltk.lemmatize.lat import LatinBackoffLemmatizer
from cltk.nlp import NLP
from cltk.tokenizers import LatinTokenizationProcess
from gensim.models import Word2Vec

cltk.data.fetch.FetchCorpus(language="lat").import_corpus("lat_models_cltk")
LEMMATIZER = LatinBackoffLemmatizer()


def is_roman_numeral(token: str) -> bool:
    """
    Tells you whether a token is a roman numeral or not

    Parameters
    ----------
    token: str
        Token to check

    Returns
    ----------
    bool
    """
    # If the length of the token is greater than zero and it is flagged
    # as a roman numeral it will return true
    return bool(token) and token.startswith("%")


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


def normalize(text: str) -> str:
    """
    Normalizes text that's given to it

    Parameters
    ----------
    text: str
        Text to normalize

    Returns
    ----------
    text: str
        Normalized text
    """
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


# Set of words that shouldn't get lemmatized
LEMMATIZATION_EXCEPTIONS = {
    "caritas",
}


def lemmatize(words: List[str]) -> List[str]:
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
    lemmata = [
        token if token in LEMMATIZATION_EXCEPTIONS else lemma
        for token, lemma in LEMMATIZER.lemmatize(words)
    ]
    bad_tags = {"punc", "-que", "e", "-ne"}
    lemmata = [
        normalize(lemma)
        for lemma in lemmata
        if lemma and not is_roman_numeral(lemma) and lemma not in bad_tags
    ]
    lemmata = remove_stopwords(lemmata)
    return lemmata


TOKENIZER = NLP(
    language="lat",
    custom_pipeline=Pipeline(
        language="lat",
        description="Tokenizer Pipeline",
        processes=[LatinTokenizationProcess],
    ),
)


def tokenize(text: str) -> List[str]:
    """
    Tokenizes the text with CLTK's tokenizer pipeline

    Parameters
    ----------
    text: str
        Text to tokenize

    Returns
    ----------
    tokens: list of str
        List of tokens in the text
    """
    return TOKENIZER.analyze(text=text).tokens


def prepare_seeds(model: Word2Vec, seeds: List[str]) -> List[str]:
    """
    Prepares the list of seeds for graph construction.
    Lemmatizes them, makes them lowercase and checks if they are in the
    Word2vec model's vocabulary.

    Parameters
    ----------
    model: Word2Vec
        Word2Vec model, that will be used for graph construction
    seeds: list of str
        Seeds to prepare

    Returns
    ----------
    seeds: list of str
        Prepared seeds
    """
    seeds = [seed.lower() for seed in seeds]
    return [seed for seed in lemmatize(seeds) if seed in model.wv.key_to_index]
