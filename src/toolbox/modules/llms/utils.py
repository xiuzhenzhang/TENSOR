from urllib.parse import urljoin
from functools import reduce


def slash_join(*args):
    return reduce(urljoin, args).rstrip("/")