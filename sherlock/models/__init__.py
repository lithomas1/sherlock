from dataclasses import dataclass
from typing import NamedTuple, List, Union
import torch
import torch.nn as nn
import torch.nn.functional as F
from nltk import sent_tokenize, word_tokenize
from operator import attrgetter
import functools

from sherlock.data.util import sanitize_text, tokenize_word
from sherlock.data.politifact import Politifact, PolFactCheck
import numpy as np


class FactCheck(NamedTuple):
    evidence: str
    strength: float


@dataclass
class Verification:
    agree: List[FactCheck]
    disagree: List[FactCheck]


class BaseModel(nn.Module):
    true = Politifact(True)[:50]
    false = Politifact(False)[:50]

    # TODO: maybe make the python equivalent of abstract class
    # __init__ defined by subclass
    def embed_words(self, words):
        ...

    def embed_sentence(self, sentence):
        ...

    def forward(self, x, y):
        ...

    def verify(self, claim: str) -> Verification:
        article = [a.claim for a in self.true]
        return self.__raw_verify(claim, article, 3)
        # get_claims = lambda pol: [p.claim for p in pol]
        # true = self.__raw_verify(claim, get_claims(self.true), 3)
        # false = self.__raw_verify(claim, get_claims(self.false), 3)
        # sort = functools.partial(sorted, key=attrgetter("strength"))
        # return Verification(
        #     agree=sort(true.agree + false.disagree),
        #     disagree=sort(true.disagree + false.agree),
        # )

    def __raw_verify(self, claim: str, article: Union[str, List[str]], k: int) -> Verification:
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
            sentence = sanitize_text(sentence)
            sentence = word_tokenize(sentence)
            sentence = [
                self.embed_words(torch.LongTensor(tokenize_word(word)))
                for word in sentence
                if word.isalnum()
            ]
            sentence = self.embed_sentence(torch.cat(sentence).unsqueeze(0)).squeeze(0)
            preds_list.append(F.softmax(self(claim, sentence)).detach())

        preds_list = np.stack(preds_list)

        agree_idxs = preds_list[1].argsort()[:k][::-1][0]
        disagree_idxs = preds_list[2].argsort()[:k][::-1][0]

        return Verification(
            agree=[
                FactCheck(article[idx], float(preds_list[1][0][idx]))
                for idx in agree_idxs
            ],
            disagree=[
                FactCheck(article[idx], float(preds_list[2][0][idx]))
                for idx in disagree_idxs
            ]
        )
