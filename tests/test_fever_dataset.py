import numpy as np


def test_indexing(fever_dataset):
    # Testing Nicolaj Costau Waldau which is supports
    claim, sentences, relevances = fever_dataset[0]
    assert len(sentences) == len(relevances)
    assert (relevances[7] == 1).all()


def test_clean_text(fever_dataset):
    text = "The_Ten_Commandments_-LRB-1956_film-RRB-"
    expected = "The_Ten_Commandments_1956_film"
    result = fever_dataset._sanitize_text(text)
    assert expected == result
