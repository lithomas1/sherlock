from dataclasses import dataclass
from typing import NamedTuple, Protocol, List


class FactCheck(NamedTuple):
    evidence: str
    strength: float


@dataclass
class Verification:
    agree: List[FactCheck]
    disagree: List[FactCheck]


class Verifier(Protocol):
    def verify(self, claim: str) -> Verification:
        ...


class DummyVerifier:
    def verify(self, claim: str) -> Verification:
        return Verification(
            agree=[
                FactCheck("the left something", 0.987),
                FactCheck("the right something", 0.912),
                FactCheck("another agree", 0.812),
            ],
            disagree=[
                FactCheck("a based wiki article", 0.912),
                FactCheck("another based article", 0.711),
                FactCheck("yipee another", 0.614),
            ]
        )
