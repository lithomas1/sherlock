import numpy as np
from sherlock.data.util import sanitize_text


def test_indexing(fever_dataset):
    claim, sentences, relevances = fever_dataset[0]
    assert len(sentences) == len(relevances)
    # assert (relevances[7] == 1).all(), relevances


def test_clean_text(fever_dataset):
    text = "The_Ten_Commandments_-LRB-1956_film-RRB-"
    expected = "The_Ten_Commandments_1956_film"
    result = sanitize_text(text)
    assert expected == result
