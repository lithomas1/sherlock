from datetime import datetime

from sherlock.data.politifact import PolFactCheck, Politifact


def test_polfactcheck_init():
    djson = {
        "speaker": "Mike Gallagher",
        "date": "December 7, 2020",
        "source": "tweet",
        "fact_checker": "Eric Litke",
        "fact_check_date": "December 18, 2020",
        "claim": "New data something"
    }
    date_items = {"date", "fact_check_date"}
    expected = {k: v for k, v in djson.items() if k not in date_items}
    expected["date"] = datetime(2020, 12, 7)
    expected["fact_check_date"] = datetime(2020, 12, 18)
    polfactcheck = PolFactCheck.from_pol_json(**djson)
    for attr in expected:
        assert getattr(polfactcheck, attr) == expected[attr], f"{attr} doesn't match"


def test_politifact_init():
    polfact = Politifact(True)
    assert len(polfact), "polfact shouldn't be empty"
    assert Politifact(True) == polfact
    assert Politifact(False) != polfact


def test_politifact_indexing():
    polfact = Politifact(True)

    assert isinstance(polfact[:30], Politifact)
    assert isinstance(polfact[0], PolFactCheck)
    assert len(polfact[:30]) == 30
