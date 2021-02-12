from datetime import datetime
from pathlib import Path
from typing import Iterable, List, NamedTuple, overload

import ujson

class PolFactCheck(NamedTuple):
    speaker: str
    date: datetime
    source: str
    fact_checker: str
    fact_check_date: datetime
    claim: str

    @classmethod
    def from_pol_json(cls, date: str, fact_check_date: str, **kwargs) -> "PolFactCheck":
        ddate = cls.str_to_date(date)
        dfc_data = cls.str_to_date(fact_check_date)
        return cls(date=ddate, fact_check_date=dfc_data, **kwargs)

    @staticmethod
    def str_to_date(string: str) -> datetime:
        return datetime.strptime(string, "%B %d, %Y")


class Politifact(List[PolFactCheck]):
    __DIR = Path(__file__).parent / "../../data/politifact"

    def __init__(self, correct: bool, existing: List[PolFactCheck] = []):
        """
        Constructor.

        :param correct: whether to search in the true claims set
        """
        if existing:
            super().__init__(existing)
        else:
            super().__init__()

        if correct:
            path = self.__DIR / "recent_true.json"
        else:
            path = self.__DIR / "recent_pants_fire.json"
        self._path = path
        self._correct = correct

        if not existing:
            self.__load_data()

    def __load_data(self):
        with open(self._path, 'r') as fin:
            self.extend(
                PolFactCheck.from_pol_json(**f)
                for f in ujson.load(fin)
            )

    @overload
    def __getitem__(self, index: int) -> PolFactCheck: ...

    @overload
    def __getitem__(self, index: slice) -> "Politifact": ...

    def __getitem__(self, index):
        if isinstance(index, slice):
            return Politifact(self._correct, super().__getitem__(index))
        return super().__getitem__(index)
