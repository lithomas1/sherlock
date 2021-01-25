import numpy as np


def test_indexing(fever_dataset):
    # Testing Nicolaj Costau Waldau which is supports
    claim, sentences, relevances = fever_dataset[0]
    assert len(sentences) == len(relevances)
    assert (relevances[7] == np.array([0, 1, 0])).all()