import torch
import torch.nn as nn
import torch.nn.functional as F

import sherlock.conf as conf
from sherlock.data.util import tokenize_word


class LSTMModel(nn.Module):
    """ A simple LSTM Model for Natural Language Inference"""

    def __init__(self, char_embeds_size=10, word_embeds_size=20, sent_embeds_size=30):
        """

        :param char_embeds_size: int, default 10
            Length of embedding vector for each character
        :param word_embeds_size: int, default 50
            Length of embedding vector for each word
        :param sent_embeds_size: int, default 100
        """
        super(LSTMModel, self).__init__()
        # 26 Character + 10 digits + Unknown Character
        # 10 dim vector for each
        self.char_embeds = nn.Embedding(37, char_embeds_size)
        self.word_lstm = nn.LSTM(
            input_size=char_embeds_size, hidden_size=word_embeds_size, batch_first=True
        )  # Word embeddings will be 50 dimensional
        self.sentence_lstm = nn.LSTM(
            input_size=word_embeds_size, hidden_size=sent_embeds_size, batch_first=True
        )

        # Prediction LSTM
        self.lstm1 = nn.LSTM(
            input_size=2, hidden_size=3, batch_first=True, bidirectional=False
        )

        #self.dense = nn.Linear(3, 3)

    def embed_words(self, words):
        """
        Generates a vector representation for each word
        :param words: str or List[int] or List[str] or List[List[int]]
            A word or list of words(or character tokens) to generate embeddings for
        :return: torch.Tensor
        """
        # TODO(@lithomas1): Testing for multiple words
        if isinstance(words, str):
            words = torch.Tensor(tokenize_word(words))
        elif isinstance(words[0], int):
            # We have single token
            words = torch.Tensor(words)
        elif isinstance(words[0], str):
            words = torch.Tensor([tokenize_word(word) for word in words])
        else:
            # Is list of list of tokens
            words = torch.LongTensor(words)

        words = torch.tanh(self.char_embeds(words.to(conf.device)))
        # Check batch
        if len(words.shape) == 2:
            words = words.unsqueeze(0)
        # TODO: Explore alternative methods to mean (e.g. take first, take last, etc)
        embeddings = torch.mean(
            self.word_lstm(words)[0], dim=1
        )  # We want output not hidden layers
        return embeddings

    def embed_sentence(self, sentence: torch.Tensor):
        """
        Generates a vector representation of a sentence

        :param sentence: torch.Tensor
            Batch first Tensor of word embeddings
        :return: torch.Tensor[sent_embeds_size]
        """
        #print(sentence)
        return torch.mean(self.sentence_lstm(sentence)[0], dim=1)
        #return self.sentence_lstm(sentence)[0][:, -1, :]

    def forward(self, x, y):
        """
        Returns relevance predictions for sentence vectors. Note:
        The output of this function are the raw logits that you'll need
        to softmax yourself to get probabilities.
        :param x: Claim
        :param y: Sentence
        :return: torch.Tensor[3]
            The relevance of the sentence
            Relevancy - 0 No relation, 1 Supports, 2 Refutes

        """
        combined = torch.cat([x, y]).unsqueeze(0)
        combined = combined.permute(0, 2, 1)  # This is bad but I have no choice :(
        #return self.dense(torch.mean(self.lstm1(combined)[0], dim=1))
        return torch.mean(self.lstm1(combined)[0], dim=1)
