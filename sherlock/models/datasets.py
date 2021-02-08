from sherlock.wiki_parser import WikiParser

from nltk.tokenize import sent_tokenize, word_tokenize
import numpy as np
import pandas as pd
from pathlib import Path
import torch
from torch.utils.data import Dataset
import warnings
from typing import Union, List, Tuple, Dict


class FEVERDataset(Dataset):
    """ PyTorch dataset used to train the NLI model on FEVER data"""

    def __init__(
        self,
        wiki_dir: Union[str, Path],
        train_file: Union[str, Path],
        tokenize: bool = True,
    ):
        """

        :param wiki_dir: str or pathlib.Path
            Directory of the Wikipedia dump
        :param train_file: str or pathlib.Path
            Path to the train.jsonl file in the FEVER dataset
        :param tokenize: bool
            Whether to tokenize the claim and sentences
        """
        self.wiki_parser = WikiParser(Path(wiki_dir))
        self.train_dataset = pd.read_json(train_file, lines=True)
        # For now only use verifiable claims, (TODO: maybe use other claims?)
        self.train_dataset = self.train_dataset[
            self.train_dataset["verifiable"] == "VERIFIABLE"
        ]
        self.tokenize = tokenize

    def _sanitize_text(self, text: str) -> str:
        """
        Removes strange artifacts from the FEVER Dataset

        Currently removing useless '-LRB-' and '-RRB' tags

        :param text: str or List[str]
            Text to sanitize
        :return: str
            Sanitized text.
        """
        if isinstance(text, list):
            for i, sent in enumerate(text):
                sent = sent.replace("-LRB-", "")
                sent = sent.replace("-LSB-", "")
                text[i] = sent.replace("-RRB-", "")
        else:
            text = text.replace("-LRB-", "")
            text = text.replace("-LRB-", "")
            text = text.replace("-RRB-", "")
        return text

    def tokenize_word(self, word: str) -> List[int]:
        """
        Maps each character in word to integer tokens.
        Numbers are mapped to themselves. Alphabet characters
        are mapped to 10-35
        Non alphanumeric characters are represented by an
        unknown token(id 36).

        :param word: str
            Input word to process
        :return: List[int]
            A List of character tokens

        """
        # Normalize
        word = word.lower()
        tokens = []
        for char in word:
            if char.isnumeric():
                tokens.append(int(char))
            elif char.isalpha():
                idx = ord(char) - 97 + 10  # 97 is ASCII 'a'
                tokens.append(idx + 10)
            else:
                tokens.append(36)

        return tokens

    def _generate_sentence_labels(self, relevance: int, articles_dict: Dict[str, List[int]]) -> Tuple[List[str],np.ndarray]:
        """

        :param relevance: int
            The relevancy of the sentence
            0 No relation, 1 Supports, 2 Refutes
        :param articles_dict: Dict[str, List[int]]
            A dict mapping article titles to a list containing
            relevant sentence indices.
        :return: Tuple[List[str], np.ndarray]
            A tuple containing a list of sentences from the articles and
            sentence relevancy labels for each.
        """
        sentences = []
        sentence_labels = []
        for (article_title, sentence_nums) in articles_dict.items():
            article = self.wiki_parser.get_entry(article_title).text
            article = sent_tokenize(self._sanitize_text(article))
            sentences += article
            labels = np.zeros((len(article), 3))
            for num in sentence_nums:
                if num >= len(labels):
                    warnings.warn(
                        f"Bad training data: sentence indice was {num} but there are only {len(labels)} sentences",
                        RuntimeWarning,
                    )
                    continue
                labels[num][relevance] += 1
            sentence_labels.append(labels)
        return sentences, sentence_labels

    def __getitem__(
        self, idx: int
    ) -> Tuple[List[List[int]], List[List[List[int]]], np.ndarray]:
        """

        :param idx: int
            Index of element to retrieve
        :return:
            Claim, Sentences of an article, and relevancy labels for each sentence
        """
        row = self.train_dataset.iloc[idx]
        # Dict matches article titles with relevant sentences nums
        articles_dict = dict()
        # Relevancy - 0 No relation, 1 Supports, 2 Refutes
        relevance = 1 if row["label"] == "SUPPORTS" else 2
        claim = row["claim"]
        evidences = row["evidence"]
        for annotator in evidences:
            # slice 'article title' and 'sentence num' cols
            annotator_evidences = np.asarray(annotator)[:, [-2, -1]]
            for (article_title, sentence_num) in annotator_evidences:
                if article_title not in articles_dict:
                    # Use set because we don't want duplicates
                    articles_dict.update({article_title: {int(sentence_num)}})
                else:
                    articles_dict[article_title].add(int(sentence_num))
        # Generate labels
        sentences, sentence_labels = self._generate_sentence_labels(relevance, articles_dict)
        # Tokenization
        claim = word_tokenize(claim)
        sentences = [word_tokenize(sentence) for sentence in sentences]
        for i, sentence in enumerate(sentences):
            to_append = []
            for word in sentence:
                if word.isalnum():
                    # TODO: (@lithomas1) maybe extract everything not punctuation?
                    to_append.append(self.tokenize_word(word))
            sentences[i] = to_append
        claim = [self.tokenize_word(word) for word in claim]
        sentence_labels = np.vstack(sentence_labels)
        return claim, sentences, sentence_labels

    def __len__(self) -> int:
        return len(self.train_dataset)
