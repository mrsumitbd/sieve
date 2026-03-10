"""
tests/test_deduplication.py

Integration tests for MinHash-based deduplication.
"""

import textwrap
from dataclasses import dataclass, field
from typing import Optional

import pytest

from sieve.core.deduplication import deduplicate, _tokenize_code


@dataclass
class _FuncStub:
    source_code: str
    func_name: str = "fn"
    repo: str = "owner/repo"


class TestTokenizeCode:
    def test_strips_string_literals(self):
        tokens = _tokenize_code('x = "hello world"')
        assert "STR" in tokens
        assert "hello" not in tokens

    def test_normalizes_whitespace(self):
        t1 = _tokenize_code("def  foo ( x ):")
        t2 = _tokenize_code("def foo(x):")
        assert t1 == t2

    def test_returns_list_of_strings(self):
        tokens = _tokenize_code("def add(a, b): return a + b")
        assert isinstance(tokens, list)
        assert all(isinstance(t, str) for t in tokens)


class TestDeduplicate:
    def test_empty_list_returned_unchanged(self):
        assert deduplicate([]) == []

    def test_no_duplicates_unchanged(self):
        records = [
            _FuncStub("def add(a, b): return a + b"),
            _FuncStub("def binary_search(arr, target):\n    lo, hi = 0, len(arr)-1\n    while lo <= hi:\n        mid = (lo+hi)//2\n        if arr[mid]==target: return mid\n        elif arr[mid]<target: lo=mid+1\n        else: hi=mid-1\n    return -1"),
            _FuncStub("class Node:\n    def __init__(self, value, next=None):\n        self.value = value\n        self.next = next\n    def __repr__(self):\n        return f'Node({self.value})'"),
        ]
        result = deduplicate(records, threshold=0.8)
        assert len(result) == 3

    def test_identical_code_deduplicated(self):
        code = "def add(a, b):\n    return a + b\n"
        records = [_FuncStub(code, func_name=f"fn{i}") for i in range(3)]
        result = deduplicate(records, threshold=0.8)
        assert len(result) == 1

    def test_first_occurrence_kept(self):
        code = "def add(a, b):\n    return a + b\n"
        records = [_FuncStub(code, func_name=f"fn{i}") for i in range(3)]
        result = deduplicate(records, threshold=0.8)
        assert result[0].func_name == "fn0"

    def test_near_duplicate_removed(self):
        """Rename a variable — should still be detected as near-duplicate."""
        code1 = textwrap.dedent("""
            def process(items):
                result = []
                for item in items:
                    result.append(item * 2)
                return result
        """).strip()
        code2 = textwrap.dedent("""
            def process(elements):
                output = []
                for element in elements:
                    output.append(element * 2)
                return output
        """).strip()
        records = [_FuncStub(code1, "fn1"), _FuncStub(code2, "fn2")]
        result = deduplicate(records, threshold=0.5)
        assert len(result) == 1

    def test_structurally_different_code_kept(self):
        code1 = "def bubble_sort(arr):\n    for i in range(len(arr)):\n        for j in range(i): arr[j], arr[j+1] = arr[j+1], arr[j]\n"
        code2 = "def binary_search(arr, target):\n    lo, hi = 0, len(arr)-1\n    while lo <= hi:\n        mid = (lo+hi)//2\n        if arr[mid] == target: return mid\n        elif arr[mid] < target: lo = mid+1\n        else: hi = mid-1\n    return -1\n"
        records = [_FuncStub(code1, "fn1"), _FuncStub(code2, "fn2")]
        result = deduplicate(records, threshold=0.8)
        assert len(result) == 2

    def test_empty_source_code_kept(self):
        """Records with empty source_code should always be retained."""
        records = [_FuncStub("", f"fn{i}") for i in range(3)]
        result = deduplicate(records, threshold=0.8)
        assert len(result) == 3

    def test_threshold_affects_aggressiveness(self):
        code1 = "def add(a, b): return a + b"
        code2 = "def add(x, y): return x + y"
        records = [_FuncStub(code1, "fn1"), _FuncStub(code2, "fn2")]
        # Low threshold — should catch minor variation
        result_low = deduplicate(records, threshold=0.3)
        # High threshold — may not catch it
        result_high = deduplicate(records, threshold=0.95)
        # Low threshold should remove more
        assert len(result_low) <= len(result_high)