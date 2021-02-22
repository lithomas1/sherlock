import numpy as np
from sherlock.data.datasets import FEVERDataset
from sherlock.models.lstm import LSTMModel
import torch
import torch.nn as nn
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
model = LSTMModel()
lr = 1e-3

# Seed for reproducibility
if seed is not None:
    np.random.seed(seed)
    torch.manual_seed(seed)

dataset = FEVERDataset("data/fever/wikidump/wiki-pages/",
                       "data/fever/train.jsonl",
                       pre_processor=model.embed_words,
                       sent_processor=model.embed_sentence)
dataloader = DataLoader(dataset,
                        shuffle=True,
                        batch_size=batch_size,
                        num_workers=num_workers)
optim = optim.Adam(model.parameters(), lr=lr)
for (claims, sentences_batch, sentence_labels) in dataloader:
    optim.zero_grad()
    # Reshaping inputs
    claims = claims.squeeze(1) # new size (1,100), assume single sentence
    sentences_batch = torch.cat(sentences_batch, dim=1)

    # Balancing classes with weighted loss
    class_weights = torch.zeros(3)
    for i in range(3):
        for sentence_label in sentence_labels:
            class_weights[i] = np.count_nonzero(sentence_label == i)
    criterion = nn.CrossEntropyLoss(weight=class_weights)
    # Generate predictions
    preds = []
    for sentences in sentences_batch:
        article_preds = []
        for sentence in sentences:
            article_preds.append(model(claims, sentence.unsqueeze(0)))
        preds.append(torch.cat(article_preds))
    preds = torch.cat(preds)
    # TODO: do this in datasets.py
    sentence_labels = torch.flatten(sentence_labels, start_dim=0, end_dim=1).long()

    loss = criterion(preds, sentence_labels)
    loss.backward()
    optim.step()
