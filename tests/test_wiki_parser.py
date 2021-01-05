from pathlib import Path

from sherlock.wiki_parser import WikiParser


def test_indexing():
    root = Path(__file__).parent.parent
    wiki = root / "data/fever/wikidump/wiki-pages/"
    wp = WikiParser(wiki)
    ids = [
        "Nikolaj_Coster-Waldau",
        "1956_SANFL_Grand_Final",
        "1996_BellSouth_Open",
    ]
    assert all(wp.get_entry(i).wid == i for i in ids)
