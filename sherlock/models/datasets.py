from sherlock.wiki_parser import WikiParser

from nltk.tokenize import sent_tokenize
import numpy as np
import pandas as pd
from pathlib import Path
import re
from torch.utils.data import Dataset


class FEVERDataset(Dataset):
    """ PyTorch dataset used to train the NLI model on FEVER data"""

    def __init__(self, wiki_dir, train_file):
        """

        :param wiki_dir: str
            Directory of the Wikipedia dump
        :param train_file: str
            Path to the train.jsonl file in the FEVER dataset
        """
        self.wiki_parser = WikiParser(Path(wiki_dir))
        self.train_dataset = pd.read_json(train_file, lines=True)
        # For now only use verifiable claims, (TODO: maybe use other claims?)
        self.train_dataset = self.train_dataset[self.train_dataset["verifiable"] == "VERIFIABLE"]

    def _sanitize_text(self, text):
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
                sent = re.sub('-LRB-', '', sent)
                text[i] = re.sub('-RRB-', '', sent)
        else:
            text = re.sub('-LRB-', '', text)
            text = re.sub('-RRB-', '', text)
        return text

    def __getitem__(self, idx):
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
        evidences = row['evidence']
        for annotator in evidences:
            # slice 'article title' and 'sentence num' cols
            annotator_evidences = np.asarray(annotator)[:, [-2, -1]]
            for (article_title, sentence_num) in annotator_evidences:
                if articles_dict.get(article_title) is None:
                    # Use set because we don't want duplicates
                    articles_dict.update({article_title: {int(sentence_num)}})
                else:
                    articles_dict[article_title].add(int(sentence_num)) # Will not append if already present
        # Generate labels
        sentences = []
        sentence_labels = []
        for (article_title, sentence_nums) in articles_dict.items():
            article = sent_tokenize(self.wiki_parser.get_entry(article_title).text)
            article = self._sanitize_text(article)
            sentences += article
            labels = np.zeros((len(article), 3))
            for num in sentence_nums:
                labels[num][relevance] += 1
            sentence_labels.append(labels)
        sentences = np.array(sentences)
        sentence_labels = np.vstack(sentence_labels)
        return claim, sentences, sentence_labels

    def __len__(self):
        return len(self.train_dataset)
