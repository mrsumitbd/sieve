"""
tests/test_discovery.py

Unit tests for GitHub search query construction.
Network is not required — _build_query is a pure function.
"""

import re
from datetime import date

import pytest

from sieve.core.discovery import _build_query


class TestBuildQuery:
    def test_pushed_range_format(self):
        q = _build_query("Python", date(2024, 1, 1), date(2025, 3, 9), 50)
        m = re.search(r"pushed:(\d{4}-\d{2}-\d{2})\.\.(\d{4}-\d{2}-\d{2})", q)
        assert m is not None
        assert m.group(1) == "2024-01-01"
        assert m.group(2) == "2025-03-09"

    def test_single_day_range(self):
        q = _build_query("Java", date(2024, 6, 15), date(2024, 6, 15), 10)
        m = re.search(r"pushed:(\d{4}-\d{2}-\d{2})\.\.(\d{4}-\d{2}-\d{2})", q)
        assert m.group(1) == m.group(2) == "2024-06-15"

    def test_stars_filter(self):
        q = _build_query("Python", date(2024, 1, 1), date(2025, 1, 1), 100)
        assert "stars:>=100" in q

    def test_mandatory_filters_always_present(self):
        q = _build_query("Python", date(2024, 1, 1), date(2025, 1, 1), 10)
        assert "fork:false" in q
        assert "archived:false" in q

    @pytest.mark.parametrize("lang,expected", [
        ("Python", "language:python"),
        ("Java", "language:java"),
        ("JavaScript", "language:javascript"),
    ])
    def test_language_mapping(self, lang, expected):
        q = _build_query(lang, date(2024, 1, 1), date(2025, 1, 1), 10)
        assert expected in q