from itertools import count
import numpy as np
import random
from sherlock.data.datasets import FEVERDataset
from sherlock.models.lstm import LSTMModel
from sherlock.models.util import collate_with_none
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader

# Download nltk punkt if not already installed
import nltk

nltk.download("punkt")

from nltk.tokenize import word_tokenize

# Ensure we are running from correct directory
import os
if "sherlock" not in {os.getcwd().split("\\")[-1], os.getcwd().split("/")[-1]}:
    raise RuntimeError("Please run this script from the base directory as python scripts/train.py")
# Config parameters
seed = 42
num_workers = 0
batch_size = 1 # debugging
model = LSTMModel().train()
lr = 1e-3
sample_ratio = np.array([0.5,1,1])

# Seed for reproducibility
if seed is not None:
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    random.seed(seed)

dataset = FEVERDataset("data/fever/wikidump/wiki-pages/",
                       "data/fever/train.jsonl",
                       pre_processor=model.embed_words,
                       sent_processor=model.embed_sentence)
dataloader = DataLoader(dataset,
                        shuffle=True,
                        batch_size=batch_size,
                        num_workers=num_workers,
                        collate_fn=collate_with_none)

optim = optim.Adam(model.parameters(), lr=lr)

def max_pred(x):
    x = list(map(float, x))
    m = max(x)
    for i, xx in enumerate(x):
        if m == xx:
            return i


def over_sample(probs, n_samples):
    return np.random.choice(np.arange(0, len(probs)), n_samples, replace=True, p=probs)


for claims, sentences_batch, sentence_labels in dataloader:

    # Check for totally corrupted data
    if len(claims) == 0:
        continue

    optim.zero_grad()

    # Balancing classes with weighted loss
    probs = np.zeros(sentence_labels.shape[1]) # TODO: fix for multi size batch
    class_weights = torch.zeros(3)
    for i in range(3):
        for sentence_label in sentence_labels:
            mask = (sentence_label == i)
            class_weights[i] = np.count_nonzero(mask)
            probs[mask] = (len(probs) / (3 * class_weights[i])) * sample_ratio[i]
    probs = probs / probs.sum()  # normalize probabilities

    # Reshaping inputs
    claims = claims.squeeze(1) # new size (1,100), assume single sentence
    sampled_idxs = over_sample(probs, n_samples=len(probs))
    sentences_batch = torch.cat(sentences_batch, dim=1)
    sentences_batch = sentences_batch[:, sampled_idxs, :]

    criterion = nn.CrossEntropyLoss()

    # Generate predictions
    preds = []
    for sentences in sentences_batch:
        article_preds = []
        for sentence in sentences:
            article_preds.append(model(claims, sentence.unsqueeze(0)))
        preds.append(torch.cat(article_preds))
    preds = torch.cat(preds)
    # TODO: do this in datasets.py
    sentence_labels = torch.flatten(
        sentence_labels, start_dim=0, end_dim=1
    ).long()[sampled_idxs]

    loss = criterion(preds, sentence_labels)
    loss.backward()
    optim.step()

    print(F.softmax(preds))
    print("Predictions:", *map(max_pred, preds))
    print("Labels:     ", *map(int, sentence_labels))
    print("Loss:       ", loss.item())
    print()
