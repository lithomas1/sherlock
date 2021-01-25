import pytest
from sherlock.models.datasets import FEVERDataset

@pytest.fixture
def fever_dataset():
    data = FEVERDataset("data/fever/wikipedia/wiki-pages/", "data/fever/train.jsonl")
    return data