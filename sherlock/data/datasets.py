from sherlock.data.wiki_parser import WikiParser
from sherlock.data.util import tokenize_word

from nltk.tokenize import sent_tokenize, word_tokenize
import numpy as np
import pandas as pd
from pathlib import Path
import unicodedata
import torch
from torch.utils.data import Dataset
import warnings
from typing import Union, List, Tuple, Dict, Callable, Optional


import random
from torch.utils.data.sampler import Sampler

class ClusterRandomSampler(Sampler):
    r"""Takes a dataset with cluster_indices property, cuts it into batch-sized chunks
    Drops the extra items, not fitting into exact batches
    Arguments:
        data_source (Dataset): a Dataset to sample from. Should have a cluster_indices property
        batch_size (int): a batch size that you would like to use later with Dataloader class
        shuffle (bool): whether to shuffle the data or not
    """

    def __init__(self, data_source, batch_size, shuffle=True):
        self.data_source = data_source
        self.batch_size = batch_size
        self.shuffle = shuffle
        
    def flatten_list(self, lst):
        return [item for sublist in lst for item in sublist]

    def __iter__(self):

        batch_lists = []
        for cluster_indices in filter(bool, self.data_source):
            batches = [cluster_indices[i:i + self.batch_size] for i in range(0, len(cluster_indices), self.batch_size)]
            # filter our the shorter batches
            batches = [_ for _ in batches if len(_) == self.batch_size]
            if self.shuffle:
                random.shuffle(batches)
            batch_lists.append(batches)       
        
        # flatten lists and shuffle the batches if necessary
        # this works on batch level
        lst = self.flatten_list(batch_lists)
        if self.shuffle:
            random.shuffle(lst)
        # final flatten  - produce flat list of indexes
        lst = self.flatten_list(lst)        
        return iter(lst)

    def __len__(self):
        return len(self.data_source)


class FEVERDataset(Dataset):
    """ PyTorch dataset used to train the NLI model on FEVER data"""

    def __init__(
        self,
        wiki_dir: Union[str, Path],
        train_file: Union[str, Path],
        tokenize: bool = True,
        pre_processor: Optional[Callable] = None,
        sent_processor: Optional[Callable] = None
    ):
        """

        :param wiki_dir: str or pathlib.Path
            Directory of the Wikipedia dump
        :param train_file: str or pathlib.Path
            Path to the train.jsonl file in the FEVER dataset
        :param tokenize: bool
            Whether to tokenize the claim and sentences
        :param pre_processor: Callable
            Function to apply to each word after tokenization.
            Takes in a torch.Tensor of tokens of each word
        :param sent_processor: Callable
            Function to apply to each sentence after tokenization and pre_processor
        """
        self.wiki_parser = WikiParser(Path(wiki_dir))
        self.train_dataset = pd.read_json(train_file, lines=True)
        # For now only use verifiable claims, (TODO: maybe use other claims?)
        self.train_dataset = self.train_dataset[
            self.train_dataset["verifiable"] == "VERIFIABLE"
        ]
        self.tokenize = tokenize
        self.pre_processor = pre_processor
        self.sent_processor = sent_processor

    def _sanitize_text(self, text: str) -> str:
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
                sent = unicodedata.normalize('NFKD', sent)
                text[i] = u"".join([c for c in text if not unicodedata.combining(c)])
        else:
            text = text.replace("-LRB-", "")
            text = text.replace("-LRB-", "")
            text = text.replace("-RRB-", "")
            # Accent removing (see comment above)
            text = unicodedata.normalize("NFKD", text)
            text = u"".join([c for c in text if not unicodedata.combining(c)])

        return text

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
            try:
                entry = self.wiki_parser.get_entry(article_title)
                article = entry.text
            except KeyError:
                warnings.warn(
                    f"Bad article title ({article_title}) found in training data",
                    RuntimeWarning)
                continue
            t_article = sent_tokenize(self._sanitize_text(article))
            article = list(map(word_tokenize,
                           map(self._sanitize_text, entry.lines)))
            sentences += article
            labels = np.zeros(len(article))
            for num in sentence_nums:
                if num >= len(labels):
                    print(len(entry.lines))
                    breakpoint()
                    warnings.warn(
                        f"Bad training data: sentence indice was {num} but there are only {len(labels)} sentences",
                        RuntimeWarning,
                    )
                    continue
                labels[num] = relevance
            sentence_labels.append(labels)
        return sentences, sentence_labels

    def __getitem__(
        self, idx: int
    ) -> Optional[Tuple[List[List[int]], List[List[List[int]]], np.ndarray]]:
        """

        :param idx: int
            Index of element to retrieve
        :return:
            Claim, Sentences of an article, and relevancy labels for each sentence
            if the training data is valid
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
        if len(sentences) == 0:
            # Data is bad
            return None
        # Tokenization/Preprocess
        claim = word_tokenize(claim)
        # try:
        #     sentences = [word_tokenize(sentence) for sentence in sentences]
        # except Exception as e:
        #     breakpoint()
        #     raise e
        # Process claim (Assume is 1 sentence)
        if self.pre_processor is not None:
            claim = [self.pre_processor(torch.LongTensor(tokenize_word(word))) for word in claim if word.isalnum()]
        if self.sent_processor is not None:
            claim = self.sent_processor(torch.cat(claim).unsqueeze(0)).squeeze(0)
        # Process sentence
        word_process = self.pre_processor is not None or self.tokenize
        for i, sentence in enumerate(sentences):
            to_append = []
            if word_process:
                # assert sentence, f"Got empty sentence"
                for word in sentence:
                    if word.isalnum():
                        # TODO: (@lithomas1) maybe extract everything not punctuation?
                        tokens = word
                        if self.tokenize:
                            tokens = torch.LongTensor(tokenize_word(word))
                        if self.pre_processor is not None:
                            tokens = self.pre_processor(tokens)
                        to_append.append(tokens)
            to_append.append(self.pre_processor(torch.LongTensor(tokenize_word("0"))))
            # assert to_append, f"Got nothing in to_append, {word_process=} {sentence=}"
            if self.sent_processor is not None:
                to_append = self.sent_processor(torch.cat(to_append).unsqueeze(0)).squeeze(0)
            sentences[i] = to_append

        sentence_labels = np.concatenate(sentence_labels)
        # print("Sentence sizes:", *{s.size() for s in sentences})
        # print("Claim size:", claim.size())
        # print("Sentence labels:", sentence_labels.size)
        return claim, sentences, sentence_labels

    def __len__(self) -> int:
        return len(self.train_dataset)
