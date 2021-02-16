import torch
import torch.nn as nn
import torch.nn.functional as F


class LSTMModel(nn.Module):
    """ A simple LSTM Model for Natural Language Inference"""

    def __init__(self, char_embeds_size=10, word_embeds_size=50, sent_embeds_size=100):
        """

        :param char_embeds_size: int, default 10
            Length of embedding vector for each character
        :param word_embeds_size: int, default 50
            Length of embedding vector for each word
        :param sent_embeds_size: int, default 100
        """
        super(CharLSTM, self).__init__()
        # 26 Character + 10 digits + Unknown Character
        # 10 dim vector for each
        self.char_embeds = nn.Embedding(37, char_embeds_size)
        self.word_lstm = nn.LSTM(
            input_size=char_embeds_size, hidden_size=word_embeds_size, batch_first=True
        )  # Word embeddings will be 50 dimensional
        self.sentence_lstm = nn.LSTM(
            input_size=word_embeds_size, hidden_size=sent_embeds_size, batch_first=True
        )
        self.dense = nn.Linear(3, 3)

    def embed_words(self, words):
        """
        Generates a vector representation for each word
        :param words: str or List[int] or List[str] or List[List[int]]
            A word or list of words(or character tokens) to generate embeddings for
        :return: List[torch.Tensor]
        """
        if isinstance(words, str):
            words = list(words)
        elif isinstance(words[0], int):
            # We have single token
            words =
        word_vectors = []
        for word in words:
            word_vectors.append(self.word_lstm(self.char_embeds(word))[0]) # We want output not hidden layers
        return word_vectors

    def forward(self, x, y):
        """
        Returns relevance predictions for sentence vectors
        :param x: Claim
        :param y: Sentence
        :return: torch.Tensor[3]
            The relevance of the sentence
            Relevancy - 0 No relation, 1 Supports, 2 Refutes
        """
        return self.
