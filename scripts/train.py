import os
import random
from itertools import count

import nltk
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from nltk.tokenize import word_tokenize
from torch.utils.data import DataLoader

from sherlock.conf import device
from sherlock.data.datasets import FEVERDataset
from sherlock.models.lstm import LSTMModel
from sherlock.models.util import collate_with_none

# Download nltk punkt if not already installed
nltk.download("punkt")

# Ensure we are running from correct directory
if "sherlock" not in {os.getcwd().split("\\")[-1], os.getcwd().split("/")[-1]}:
    raise RuntimeError(
        "Please run this script from the base directory as python scripts/train.py"
    )

# Config parameters
seed = 42
num_workers = 0
batch_size = 1  # debugging
model = LSTMModel().train().to(device)
lr = 1e-3
sample_ratio = np.array([0.5, 1, 1])
model_path = "checkpoints/{}/model.pt"
optimizer_path = "checkpoints/{}/adam.pt"
try:
    checkpoints = sorted(map(int, os.listdir("checkpoints")))
    c_num = len(checkpoints)
except FileNotFoundError:
    os.mkdir("checkpoints")
    checkpoints = []
    c_num = 0

# Seed for reproducibility
if seed is not None:
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    random.seed(seed)

dataset = FEVERDataset(
    "data/fever/wikidump/wiki-pages/",
    "data/fever/train.jsonl",
    pre_processor=model.embed_words,
    sent_processor=model.embed_sentence,
)
dataloader = DataLoader(
    dataset,
    shuffle=True,
    batch_size=batch_size,
    num_workers=num_workers,
    collate_fn=collate_with_none,
)

optim = optim.Adam(model.parameters(), lr=lr)

if c_num:
    model.load_state_dict(model_path.format(checkpoints[-1]))
    optim.load_state_dict(optimizer_path.format(checkpoints[-1]))

def save_checkpoint():
    global c_num
    c_num += 1
    os.mkdir(f"checkpoints/{c_num}")
    torch.save(model.state_dict(), model_path.format(c_num))
    torch.save(optim.state_dict(), optimizer_path.format(c_num))

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
    probs = np.zeros(sentence_labels.shape[1])  # TODO: fix for multi size batch
    class_weights = torch.zeros(3)
    for i in range(3):
        for sentence_label in sentence_labels:
            mask = sentence_label == i
            class_weights[i] = np.count_nonzero(mask)
            probs[mask] = (len(probs) / (3 * class_weights[i])) * sample_ratio[i]
    probs = probs / probs.sum()  # normalize probabilities

    # Reshaping inputs
    claims = claims.squeeze(1)  # new size (1,100), assume single sentence
    sampled_idxs = over_sample(probs, n_samples=len(probs))
    sentences_batch = torch.cat(sentences_batch, dim=1)
    sentences_batch = sentences_batch[:, sampled_idxs, :]

    criterion = nn.CrossEntropyLoss()

    claims = claims.to(device)
    sentence_labels = sentence_labels.to(device)

    # Generate predictions
    preds = []
    for sentences in sentences_batch:
        article_preds = []
        for sentence in sentences:
            sentence = sentence.to(device)
            article_preds.append(model(claims, sentence.unsqueeze(0)))
        preds.append(torch.cat(article_preds))
    preds = torch.cat(preds)
    # TODO: do this in datasets.py
    sentence_labels = torch.flatten(sentence_labels, start_dim=0, end_dim=1).long()[
        sampled_idxs
    ]

    loss = criterion(preds, sentence_labels)
    loss.backward()
    optim.step()

    print(F.softmax(preds))
    print("Predictions:", *map(max_pred, preds))
    print("Labels:     ", *map(int, sentence_labels))
    print("Loss:       ", loss.item())
    print()
