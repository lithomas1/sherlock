"""
Tool to scrape claims from politifact.

Requires Python 3.8 or greater to run due to asyncio.
"""

import argparse
import asyncio
import dataclasses
import functools
import itertools as it
import json
import re
from typing import ClassVar, List, Optional

import aiohttp
from bs4 import BeautifulSoup, Tag


def suppress_exceptions(default=None):
    def wrapper(func):
        @functools.wraps(func)
        def inner(*args, **kwargs):
            try:
                return func(*args, **kwargs)
            except:
                return default
        return inner
    return wrapper


@dataclasses.dataclass
class FactCheck:
    speaker: str
    date: str
    source: str  # interview, twitter, etc
    fact_checker: str
    fact_check_date: str
    claim: str

    description_date: ClassVar[re.Pattern] = re.compile(
        r"^stated on ([a-zA-Z]+ \d+, \d+)"
    )
    description_source: ClassVar[re.Pattern] = re.compile(r"in a?n?\ ?(.+):$")
    footer_author: ClassVar[re.Pattern] = re.compile(r"^By (.+) •")
    footer_date: ClassVar[re.Pattern] = re.compile(r"• (.+)$")

    @classmethod
    @suppress_exceptions()
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
    @suppress_exceptions()
    def speaker_from_card(card: Tag) -> str:
        return card.find("a", {"class": "m-statement__name"}).text.strip()

    @classmethod
    @suppress_exceptions()
    def date_from_card(cls, card: Tag) -> str:
        description = card.find("div", {"class": "m-statement__desc"}).text.strip()
        return re.search(cls.description_date, description).group(1).strip()

    @classmethod
    @suppress_exceptions()
    def source_from_card(cls, card: Tag) -> str:
        description = card.find("div", {"class": "m-statement__desc"}).text.strip()
        return re.search(cls.description_source, description).group(1).strip()

    @classmethod
    @suppress_exceptions()
    def fact_checker_from_card(cls, card: Tag) -> str:
        footer = card.find("footer", {"class": "m-statement__footer"}).text.strip()
        return re.search(cls.footer_author, footer).group(1).strip()

    @classmethod
    @suppress_exceptions()
    def fact_check_date_from_card(cls, card: Tag) -> str:
        footer = card.find("footer", {"class": "m-statement__footer"}).text.strip()
        return re.search(cls.footer_date, footer).group(1).strip()

    @staticmethod
    def claim_from_card(card: Tag) -> str:
        quote = card.find("div", {"class": "m-statement__quote"})
        return quote.text.strip().replace("\u201c", "").replace("\u201d", "")


@suppress_exceptions([])
def scrape_facts_list(html: str) -> List[Optional[FactCheck]]:
    soup = BeautifulSoup(html, "html.parser")
    cards = soup.find_all("li", {"class": "o-listicle__item"})
    return list(map(FactCheck.from_card, cards))


async def scrape_facts_from_page(
    page: int, ruling: str, session: aiohttp.ClientSession
) -> List[Optional[FactCheck]]:
    """
    Scrape the fact checks from website.

    :param page: the page number greater than 1
    :param ruling: the verdict on the fact check eg. true, pants-fire
    :param session: the aiohttp session to use
    :return: the fact checks on the page
    """
    r = await session.request(
        method="GET",
        url=f"https://www.politifact.com/factchecks/list/?page={page}&ruling={ruling}",
    )
    return scrape_facts_list(await r.text())


async def main():
    parser = argparse.ArgumentParser(description="Politifact fact check scraper")
    parser.add_argument("-s", "--page-start", type=int, default=1)
    parser.add_argument("-e", "--page-end", type=int, default=20)
    parser.add_argument("-r", "--ruling", type=str, default="true")
    args = parser.parse_args()
    async with aiohttp.ClientSession() as session:
        checks = await asyncio.gather(
            *(scrape_facts_from_page(i, args.ruling, session) for i in range(args.page_start, args.page_end))
        )
    checks = [cc for cc in it.chain(*(c for c in checks if c)) if cc]
    v[:] = checks
    print(json.dumps(list(map(dataclasses.asdict, checks)), indent=2))


if __name__ == "__main__":
    # file_main()
    v = []
    asyncio.run(main())
