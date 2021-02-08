import pytest
from sherlock.models.datasets import FEVERDataset

@pytest.fixture(scope="module")
def fever_dataset():
    data = FEVERDataset("data/fever/wikidump/wiki-pages/", "data/fever/train.jsonl")
    return data