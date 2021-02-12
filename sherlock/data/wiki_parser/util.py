import re
from pathlib import Path


def num_in_path(path_with_num: Path) -> int:
    return num_in_str(str(path_with_num))


def num_in_str(string_with_num: str) -> int:
    try:
        return int(re.search(r"(\d+)", string_with_num).group(1))
    except (AttributeError, ValueError):
        return -1
