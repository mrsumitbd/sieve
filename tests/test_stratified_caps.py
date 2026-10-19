"""
tests/test_stratified_caps.py

Unit tests for _stratified_caps (module-level in pipeline.py).
"""

import pytest

from sieve.pipeline import _stratified_caps


class TestStratifiedCapsBasics:
    @pytest.mark.parametrize("counts", [
        {"a": 10, "b": 5},
        {"a": 100, "b": 50, "c": 25, "d": 25},
        {"a": 3, "b": 3, "c": 3},
    ])
    def test_zero_cap_gives_all_zeros(self, counts):
        assert _stratified_caps(counts, 0) == {r: 0 for r in counts}

    @pytest.mark.parametrize("counts,total_cap", [
        ({"a": 10, "b": 5}, 5),
        ({"a": 10, "b": 5}, 7),
        ({"a": 10, "b": 5}, 10),
        ({"a": 100, "b": 50, "c": 25, "d": 25}, 10),
        ({"a": 100, "b": 50, "c": 25, "d": 25}, 20),
        ({"a": 3, "b": 3, "c": 3}, 7),
    ])
    def test_sum_matches_total_cap(self, counts, total_cap):
        caps = _stratified_caps(counts, total_cap)
        assert set(caps) == set(counts)
        assert sum(caps.values()) == total_cap

    def test_empty_counts_dict(self):
        assert _stratified_caps({}, 5) == {}

    def test_all_zero_counts(self):
        assert _stratified_caps({"a": 0, "b": 0}, 5) == {"a": 0, "b": 0}
