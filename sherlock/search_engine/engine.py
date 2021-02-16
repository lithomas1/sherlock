from pathlib import Path
from sherlock.wiki_parser import WikiParser


class SearchEngine:
    """
    A search engine that returns the top k results
    matching a query
    """

    def __init__(self, data_dir):
        """
        :param data_dir: str
            The path to the directory of articles
        """
        self.data_dir = data_dir
        self.wiki_parser = WikiParser(Path(data_dir))

    def get_results(self, query, k=5):
        """

        :param query: str
            Query to search for
        :param k: int {default: 5}
            Specifies the number of results to return
        :return: List of results
        """
