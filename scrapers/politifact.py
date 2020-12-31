"""
Tool to scrape claims from politifact.

Requires Python 3.8 or greater to run due to asyncio.
"""

import asyncio
import json
import re
import functools
import dataclasses
from typing import ClassVar, List, Optional

import aiohttp
from bs4 import BeautifulSoup, Tag


def suppress_exceptions(func):
    @functools.wraps(func)
    def inner(*args, **kwargs):
        try:
            return func(*args, **kwargs)
        except:
            return
    return inner


@dataclasses.dataclass
class FactCheck:
    speaker: str
    date: str
    source: str  # interview, twitter, etc
    fact_checker: str
    fact_check_date: str
    claim: str

    description_date: ClassVar[re.Pattern] = re.compile(r"^stated on ([a-zA-Z]+ \d+, \d+)")
    description_source: ClassVar[re.Pattern] = re.compile(r"in a?n?\ ?(.+):$")
    footer_author: ClassVar[re.Pattern] = re.compile(r"^By (.+) •")
    footer_date: ClassVar[re.Pattern] = re.compile(r"• (.+)$")

    @classmethod
    @suppress_exceptions
    def from_card(cls, card: Tag):
        return cls(
            speaker=cls.speaker_from_card(card),
            date=cls.date_from_card(card),
            source=cls.source_from_card(card),
            fact_checker=cls.fact_checker_from_card(card),
            fact_check_date=cls.fact_check_date_from_card(card),
            claim=cls.claim_from_card(card),
        )

    def to_json(self) -> str:
        return dataclasses.asdict(self)

    @staticmethod
    def speaker_from_card(card: Tag) -> str:
        return card.find("a", {"class": "m-statement__name"}).text.strip()

    @classmethod
    def date_from_card(cls, card: Tag) -> str:
        description = card.find("div", {"class": "m-statement__desc"}).text.strip()
        return re.search(cls.description_date, description).group(1).strip()

    @classmethod
    def source_from_card(cls, card: Tag) -> str:
        description = card.find("div", {"class": "m-statement__desc"}).text.strip()
        return re.search(cls.description_source, description).group(1).strip()

    @classmethod
    def fact_checker_from_card(cls, card: Tag) -> str:
        footer = card.find("footer", {"class": "m-statement__footer"}).text.strip()
        return re.search(cls.footer_author, footer).group(1).strip()

    @classmethod
    def fact_check_date_from_card(cls, card: Tag) -> str:
        footer = card.find("footer", {"class": "m-statement__footer"}).text.strip()
        return re.search(cls.footer_date, footer).group(1).strip()

    @staticmethod
    def claim_from_card(card: Tag) -> str:
        quote = card.find("div", {"class": "m-statement__quote"})
        return quote.text.strip().replace("\u201c", "").replace("\u201d", "")


def scrape_facts_list(html: str) -> List[Optional[FactCheck]]:
    soup = BeautifulSoup(html, "html.parser")
    cards = soup.find_all("li", {"class": "o-listicle__item"})
    return list(map(FactCheck.from_card, cards))


def file_main():
    with open("politifact.html", 'r') as fin:
        html = fin.read()
    print(json.dumps([*map(FactCheck.to_json, scrape_facts_list(html))], indent=2))


async def main():
    ...


if __name__ == "__main__":
    file_main()
    asyncio.run(main())
