import math
from functools import reduce
from operator import add, mul, sub, truediv

from src.toolbox.list_operation.framework import apply_ops_on_list1


def list_mean(lst):
    return reduce(add, lst) / len(lst)


def list_std(lst):
    x2 = [x * x for x in lst]
    return math.sqrt(list_mean(x2) - list_mean(lst) ** 2)


def list_add(list1, second_input):
    return apply_ops_on_list1(list1, second_input, add)


def list_sub(list1, second_input):
    return apply_ops_on_list1(list1, second_input, sub)


def list_mul(list1, multipler):
    return apply_ops_on_list1(list1, multipler, mul)


def list_div(list1, denominator):
    return apply_ops_on_list1(list1, denominator, truediv)
