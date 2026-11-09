"""Module docstring."""
import os
import functools
from dataclasses import dataclass


def top_level(a, b=2, *args, **kwargs):
    """Top-level function."""
    def inner(x):
        return x + a

    class LocalInFunction:
        def method(self):
            return os.sep

    return inner(b)


async def fetch(url: str) -> bytes:
    async def helper():
        return b""
    return await helper()


@functools.lru_cache(maxsize=None)
def cached(n):
    return n


@dataclass(frozen=True)
class Point:
    x: int
    y: int

    @property
    def norm(self):
        return (self.x ** 2 + self.y ** 2) ** 0.5

    @staticmethod
    def origin():
        return Point(0, 0)


class Base(object, metaclass=type):
    """Base class."""

    def __init__(self):
        self.v = 1

    class Nested:
        def deep(self):
            def deeper():
                return 1
            return deeper

    async def run(self):
        return [lambda: 1 for _ in range(3)]


if os.name == "nt":
    def windows_only():
        return 1
else:
    def posix_only():
        return 2

try:
    import json
except ImportError:
    def fallback():
        return None
