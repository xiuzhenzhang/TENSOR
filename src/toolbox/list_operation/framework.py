import numbers


def apply_ops_on_list1(list1, second_value, ops):
    if isinstance(second_value, numbers.Number):
        return [ops(x, second_value) for x in list1]

    return [ops(x, y) for x, y in zip(list1, second_value)]
