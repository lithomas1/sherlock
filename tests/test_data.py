from sherlock.data.wiki_parser import data


def test_wikientry_from_json():
    iid = "1928_in"
    text = "The following are the football -LRB- soccer -RRB- events of the year 1928 throughout the world ."
    lines = "0\tThe following are the football -LRB- soccer -RRB- events of the year 1928 throughout the world .\n1\tNext sentence\n2\t"
    correct_lines = [
        "The following are the football -LRB- soccer -RRB- events of the year 1928 throughout the world .",
        "Next sentence",
    ]
    m = data.WikiEntry.from_wiki_json(
        {
            "id": iid,
            "text": text,
            "lines": lines,
        }
    )
    assert m.lines == correct_lines
