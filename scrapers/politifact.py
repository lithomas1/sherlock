"""
Tool to scrape claims from politifact.

Requires Python 3.8 or greater to run due to asyncio.
"""

import asyncio
from dataclasses import dataclass
from typing import List, Optional

import aiohttp


@dataclass
class FactCheck:
    speaker: str
    date: str
    source: str  # interview, twitter, etc
    fact_checker: str
    fact_check_date: str
    claim: str


def scrape_facts_list(html: str) -> List[Optional[FactCheck]]:
    ...


async def main():
    ...


if __name__ == "__main__":
    asyncio.run(main())
