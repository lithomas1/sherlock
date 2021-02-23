import re
from dataclasses import dataclass
from typing import Dict, List


def f(x):
    return True

@dataclass
class WikiEntry:
    wid: str
    text: str
    lines: List[str]
    oglines: str

    @classmethod
    def from_wiki_json(cls, entry: Dict[str, str]):
        lines = list(filter(f, re.split(r"\n\d+\t", entry["lines"])))
        if lines[0][:2] == "0\t":
            lines[0] = lines[0][2:]
        return cls(entry["id"], entry["text"], lines, entry["lines"])
