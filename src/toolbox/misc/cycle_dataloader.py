from collections.abc import Iterable


def cycle(iterable: Iterable) -> Iterable:
    def gen_cycle():
        while True:
            yield from iterable

    return iter(gen_cycle())
