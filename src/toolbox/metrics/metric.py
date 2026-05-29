import math

from src.toolbox.list_operation import list_mul
from src.toolbox.misc import check_number


class Metric:
    """
    A Metric handler.
    1. metric_number: How many metric do you have?
    2. smaller_is_better: If model performance is better with lower metric value, you should set it to true. Otherwise, it is false.
    If smaller_is_better is set, its length must match argument 'metric_number'.
    """

    def __init__(self, metric_number, smaller_is_better=None):
        self.metric_number = metric_number
        self.map = {True: 1, False: -1}
        self.best_metric = [math.inf] * self.metric_number
        if smaller_is_better is None:
            self.mask = [1] * self.metric_number
        else:
            if len(smaller_is_better) != self.metric_number:
                raise ValueError("The length of smaller_is_better must be the same as that of self.metric_number!")
            self.mask = [self.map[item] for item in smaller_is_better]

    def compare(self, input_metric):
        if len(input_metric) != len(self.mask):
            raise ValueError("The length of input_metric must be the same as that of self.mask!")
        tmp = list_mul(input_metric, self.mask)
        result = True

        for input_number, recorded in zip(tmp, self.best_metric):
            if check_number(input_number, positive=False, break_out=False) and input_number > recorded:
                result = False
                break

        if result:
            self.best_metric = tmp

        return result

    def show(self):
        return list_mul(self.best_metric, self.mask)


if __name__ == "__main__":
    metric_number = 4
    metric = Metric(metric_number)

    first_metric = [1] * 4
    result = metric.compare(first_metric)
    assert result, "Why do not accept the first group of metric values?"
    assert metric.show() == [1, 1, 1, 1], "We stored the wrong metric values!"
    print(metric.show())

    second_metric = [0.5] * 4
    result = metric.compare(second_metric)
    assert result, "This one should be better!"
    assert metric.show() == [0.5, 0.5, 0.5, 0.5], "We stored the wrong metric values!"
    print(metric.show())

    second_metric = [0.25] * 4
    result = metric.compare(second_metric)
    assert result, "This one should be better!"
    assert metric.show() == [0.25, 0.25, 0.25, 0.25], "We stored the wrong metric values!"
    print(metric.show())

    third_metric = [0.75] * 4
    result = metric.compare(third_metric)
    assert result, "Metric judgement is wrong!"
    assert metric.show() == [0.25, 0.25, 0.25, 0.25], "We stored the wrong metric values!"
    print(metric.show())

    metric_number = 4
    metric = Metric(metric_number, smaller_is_better=[False, False, True, True])

    first_metric = [1] * 4
    result = metric.compare(first_metric)
    assert result, "Why do not accept the first group of metric values?"
    assert metric.show() == [1, 1, 1, 1], "We stored the wrong metric values!"
    print(metric.show())

    second_metric = [1.5, 1.5, 0.5, 0.5]
    result = metric.compare(second_metric)
    assert result, "This one should be better!"
    assert metric.show() == [1.5, 1.5, 0.5, 0.5], "We stored the wrong metric values!"
    print(metric.show())

    second_metric = [2.0, 2.0, 0.25, 0.25]
    result = metric.compare(second_metric)
    assert result, "This one should be better!"
    assert metric.show() == [2.0, 2.0, 0.25, 0.25], "We stored the wrong metric values!"
    print(metric.show())

    third_metric = [2.0, 2.0, 0.75, 0.75]
    result = metric.compare(third_metric)
    assert result, "Metric judgement is wrong!"
    assert metric.show() == [2.0, 2.0, 0.25, 0.25], "We stored the wrong metric values!"
    print(metric.show())

    metric_number = 1
    metric = Metric(metric_number)

    first_metric = [1] * 1
    result = metric.compare(first_metric)
    assert result, "Why do not accept the first group of metric values?"
    assert metric.show() == [
        1,
    ], "We stored the wrong metric values!"
    print(metric.show())

    second_metric = [0.5] * 1
    result = metric.compare(second_metric)
    assert result, "This one should be better!"
    assert metric.show() == [
        0.5,
    ], "We stored the wrong metric values!"
    print(metric.show())

    second_metric = [0.25] * 1
    result = metric.compare(second_metric)
    assert result, "This one should be better!"
    assert metric.show() == [
        0.25,
    ], "We stored the wrong metric values!"
    print(metric.show())

    third_metric = [0.75] * 1
    result = metric.compare(third_metric)
    assert result, "Metric judgement is wrong!"
    assert metric.show() == [
        0.25,
    ], "We stored the wrong metric values!"
    print(metric.show())
