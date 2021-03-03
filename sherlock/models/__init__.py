from dataclasses import dataclass
from typing import NamedTuple, List, Union
from pathlib import Path
import torch
import torch.nn as nn
import torch.nn.functional as F
from sherlock.conf import device
from nltk import sent_tokenize, word_tokenize
from operator import attrgetter
import functools

from sherlock.data.util import sanitize_text, tokenize_word
from sherlock.data.politifact import Politifact, PolFactCheck
import numpy as np


class FactCheck(NamedTuple):
    evidence: str
    strength: float
    is_null: float


@dataclass
class Verification:
    agree: List[FactCheck]
    disagree: List[FactCheck]


class BaseModel(nn.Module):
    true = Politifact(True)[:50]
    false = Politifact(False)[:50]
    model_path = (Path(__file__).parent / "../../models/model.pt").resolve()

    # TODO: maybe make the python equivalent of abstract class
    # __init__ defined by subclass
    def embed_words(self, words):
        ...

    def embed_sentence(self, sentence):
        ...

    def forward(self, x, y):
        ...

    def sherlock_load(self):
        if self.model_path.exists():
            self.load_state_dict(torch.load(str(self.model_path), map_location=device))
        else:
            print("[WARNING]", f"no model found at {self.model_path}")

    def verify(self, claim: str) -> Verification:
        get_claims = lambda pol: [p.claim for p in pol]
        true = self.__raw_verify(claim, get_claims(self.true), 3)
        false = self.__raw_verify(claim, get_claims(self.false), 3)
        for i, f in enumerate(false.agree):
            false.agree[i] = f._replace(evidence=f"Proven false: {f.evidence}")
        for i, f in enumerate(false.disagree):
            false.disagree[i] = f._replace(evidence=f"Proven false: {f.evidence}")
        sort = functools.partial(
            sorted,
            key=attrgetter("strength"),
            reverse=True
        )

        # We don't use false.disagree since a lot of the false claims
        # are crazy and denial of them means nothing
        return Verification(
            agree=sort(true.agree)[:3],
            disagree=sort(true.disagree + false.agree)[:3],
        )

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
        claim = self.embed_sentence(torch.cat(claim).unsqueeze(0)).squeeze(1)
        preds_list = []
        for sentence in article:
            sentence = sanitize_text(sentence)
            sentence = word_tokenize(sentence)
            sentence = [
                self.embed_words(torch.LongTensor(tokenize_word(word)))
                for word in sentence
                if word.isalnum()
            ]
            sentence = self.embed_sentence(torch.cat(sentence).unsqueeze(0)).squeeze(1)
            preds_list.append(F.softmax(self(claim, sentence)).detach())

        preds_list = np.stack(preds_list)

        l = len(preds_list)
        ids = [i for i in range(l) if max(preds_list[i][0]) != preds_list[i][0][0]]
        sort = functools.partial(sorted, ids, reverse=True)
        agree_idxs = sort(key=lambda x: preds_list[x][0][1])[:k]
        disagree_idxs = sort(key=lambda x: preds_list[x][0][2])[:k]

        return Verification(
            agree=[
                FactCheck(article[idx], *map(float, (preds_list[idx][0][:2][::-1])))
                for idx in agree_idxs
            ],
            disagree=[
                FactCheck(article[idx], *map(float, (preds_list[idx][0][::2][::-1])))
                for idx in disagree_idxs
            ]
        )
