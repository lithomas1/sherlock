from dataclasses import dataclass
from typing import NamedTuple, Protocol, List
import torch
import torch.nn as nn
from nltk import sent_tokenize, word_tokenize

from sherlock.data.util import tokenize_word
import unicodedata
import numpy as np


class FactCheck(NamedTuple):
    evidence: str
    strength: float


@dataclass
class Verification:
    agree: List[FactCheck]
    disagree: List[FactCheck]


class Verifier(Protocol):
    def verify(self, claim: str) -> Verification:
        ...


class BaseModel(nn.Module):
    # TODO: maybe make the python equivalent of abstract class
    # __init__ defined by subclass
    def embed_words(self, words):
        pass

    def embed_sentence(self, sentence):
        pass

    def forward(self, x, y):
        pass

    def _sanitize_text(self, text: str) -> str:
        # TODO: de-dup with the other one
        """
        Normalizes words and removes strange artifacts from the FEVER Dataset

        Currently removes accents and useless '-LRB-' and '-RRB' tags

        :param text: str or List[str]
            Text to sanitize
        :return: str
            Sanitized text.
        """
        if isinstance(text, list):
            for i, sent in enumerate(text):
                sent = sent.replace("-LRB-", "")
                sent = sent.replace("-LSB-", "")
                sent = sent.replace("-RRB-", "")

                # Accent removing
                # ref https://tinyurl.com/3kae7p6r (Stack Overflow)
                sent = unicodedata.normalize("NFKD", sent)
                text[i] = "".join([c for c in text if not unicodedata.combining(c)])
        else:
            text = text.replace("-LRB-", "")
            text = text.replace("-LRB-", "")
            text = text.replace("-RRB-", "")
            # Accent removing (see comment above)
            text = unicodedata.normalize("NFKD", text)
            text = "".join([c for c in text if not unicodedata.combining(c)])

        return text

    def verify(self, claim: str, article: str, k: int) -> Verification:
        """

        :param claim: str
            The claim to evaluate
        :param article: str or List[str]
            Collection of evidence to evaluate on
        :param k: int
            The number of agree and disagree evidence to return
        :return: The top k agree and disgree evidence
        """
        if isinstance(article, str):
            article = sent_tokenize(article)
        claim = word_tokenize(claim)
        claim = [
            self.embed_words(torch.LongTensor(tokenize_word(word)))
            for word in claim
            if word.isalnum()
        ]
        claim = self.embed_sentence(torch.cat(claim).unsqueeze(0)).squeeze(0)
        preds_list = []
        for sentence in article:
            sentence = word_tokenize(sentence)
            sentence = [
                self.embed_words(torch.LongTensor(tokenize_word(word)))
                for word in sentence
                if word.isalnum()
            ]
            sentence = self.embed_sentence(torch.cat(sentence).unsqueeze(0)).squeeze(0)
            preds_list.append(self.forward(claim, sentence))

        preds_list = np.stack(preds_list)

        agree_idxs = preds_list[1].argsort()[:k]
        disagree_idxs = preds_list[2].argsort()[:k]

        return Verification(
            agree=[
                FactCheck(article[idx], preds_list[1][idx])
                for idx in agree_idxs
            ],
            disagree=[
                FactCheck(article[idx], preds_list[2][idx])
                for idx in disagree_idxs
            ]
        )
