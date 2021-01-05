import gc
from pathlib import Path

import ujson
from sherlock.wiki_parser.util import num_in_path


class Indexer(dict):
    def __init__(self, file: Path):
        super().__init__()
        self._file = file
        self._fileno = num_in_path(file)
        self.__load_indeces()
        gc.collect()

    def __load_indeces(self):
        try:
            self.__load_indeces_if_exists()
        except Exception:
            self.__generate_indeces()
            self.__save_indeces()

    def __get_index_file_path(self) -> Path:
        return Path(f"{self._file}.idx")

    def __load_indeces_if_exists(self):
        with open(self.__get_index_file_path(), "r") as fin:
            self.update(ujson.load(fin))

    def __generate_indeces(self):
        with open(self._file, "r") as fin:
            try:
                while 1:
                    seek_pos = fin.tell()
                    line_contents = fin.readline()
                    wiki_id = ujson.loads(line_contents)["id"]
                    if wiki_id:
                        self[wiki_id] = (seek_pos, self._fileno)
            # Will be raised upon trying to ujson an empty string
            except ValueError:
                return

    def __save_indeces(self):
        with open(self.__get_index_file_path(), "w+") as fout:
            ujson.dump(self, fout)
