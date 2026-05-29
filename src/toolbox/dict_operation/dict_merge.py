from collections.abc import Iterable
from typing import Any


def dict_merge(input_dict: Iterable[dict[Any, Any]]) -> dict[Any, list[Any]]:
    result_dict = {}

    for item in input_dict:
        for key, value in item.items():
            if key in result_dict:
                result_dict[key].append(value)
            else:
                result_dict[key] = [value]

    return result_dict


if __name__ == "__main__":
    input_data = [{'a': 1, 'b': 4}, {'a': 2, 'b': 5}, {'a': 3, 'b': 6}]
    print(dict_merge(input_data))

    input_data = [{'a': 1, 'b': 4, 'c': 6, 'd': [11, 12]}, {'a': 2, 'c': 5}, {'a': 3, 'd': 6}]
    print(dict_merge(input_data))
