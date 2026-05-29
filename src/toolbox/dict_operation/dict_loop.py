from collections.abc import Iterable
from typing import Any


def dict_loop(input_dict: dict[Any, list]) -> Iterable:
    length_set = {len(item) for item in input_dict.values()}
    if len(length_set) != 1:
        raise ValueError("Some keys have a list with different length from other keys, which is unexpected.")

    length_of_all_list = length_set.pop()
    for idx in range(length_of_all_list):
        yield {key: input_dict[key][idx] for key in input_dict}



if __name__ == "__main__":
    input_data = {'a': [1, 2, 3], 'b': [4, 5, 6]}

    assert list(dict_loop(input_data)) == [{'a': 1, 'b': 4}, {'a': 2, 'b': 5}, {'a': 3, 'b': 6}]

    input_data = {'a': [1, 2, 3], 'b': [4, 5, 6, 7]}
    try:
        list(dict_loop(input_data))
    except ValueError as e:
        print(e)
