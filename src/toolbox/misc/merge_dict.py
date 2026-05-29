from functools import reduce
from typing import Any


def merge_list_of_dicts(input_dict: list[dict[Any, Any]]) -> list[dict[Any, Any]]:
    return reduce(dict.__or__, input_dict)
