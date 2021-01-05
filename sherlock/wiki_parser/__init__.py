import gc
from pathlib import Path
from typing import Dict

import ujson
from sherlock.wiki_parser.data import WikiEntry
from sherlock.wiki_parser.indexer import Indexer
from sherlock.wiki_parser.util import num_in_path
from tqdm import tqdm


class WikiParser:
    cache: Dict[Path, Indexer] = dict()

    def __init__(self, wikidir: Path):
        self.wikidir = wikidir
        self.wiki_paths = sorted(wikidir.glob("*.jsonl"), key=num_in_path)
        try:
            self.indexer = self.cache[wikidir.resolve()]
        except KeyError:
            print("Loading indeces")
            indexers = list(
                tqdm(map(Indexer, self.wiki_paths), total=len(self.wiki_paths))
            )
            self.indexer = indexers[0]
            if len(indexers) > 1:
                print("Merging indeces")
                for idxer in tqdm(indexers[1:]):
                    self.indexer.update(idxer)
                    gc.collect()
            self.cache[wikidir.resolve()] = self.indexer

    def get_entry(self, wiki_id: str) -> WikiEntry:
        seek_pos, fileno = self.indexer[wiki_id]
        with open(self.wiki_paths[fileno - 1], "r") as fin:
            fin.seek(seek_pos)
            entry = ujson.loads(fin.readline())
            return WikiEntry.from_wiki_json(entry)
