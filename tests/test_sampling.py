"""
tests/test_sampling.py

Unit tests for _stratified_sample (module-level in pipeline.py).
"""

from dataclasses import dataclass

import pytest

from sieve.pipeline import _stratified_sample


@dataclass
class _Rec:
    """Minimal record stub with a repo attribute."""
    repo: str
    id: int


def _make_records(repo_sizes: dict[str, int]) -> list[_Rec]:
    records = []
    counter = 0
    for repo, n in repo_sizes.items():
        for _ in range(n):
            records.append(_Rec(repo=repo, id=counter))
            counter += 1
    return records


class TestStratifiedSample:
    def test_no_op_when_under_cap(self):
        records = _make_records({"A": 5, "B": 3})
        result = _stratified_sample(records, cap=100)
        assert len(result) == 8

    def test_total_equals_cap(self):
        records = _make_records({"A": 12, "B": 6, "C": 2})
        result = _stratified_sample(records, cap=10)
        assert len(result) == 10

    def test_no_duplicates(self):
        records = _make_records({"A": 12, "B": 6, "C": 2})
        result = _stratified_sample(records, cap=10)
        ids = [r.id for r in result]
        assert len(ids) == len(set(ids))

    def test_proportional_allocation(self):
        """A=12, B=6, C=2 with cap=10: A should get ~6, B ~3, C ~1."""
        records = _make_records({"A": 12, "B": 6, "C": 2})
        result = _stratified_sample(records, cap=10)
        by_repo = {}
        for r in result:
            by_repo[r.repo] = by_repo.get(r.repo, 0) + 1
        # A should have the most, C should have the least
        assert by_repo.get("A", 0) >= by_repo.get("B", 0) >= by_repo.get("C", 0)

    def test_single_repo(self):
        records = _make_records({"A": 20})
        result = _stratified_sample(records, cap=5)
        assert len(result) == 5
        assert all(r.repo == "A" for r in result)

    def test_equal_sized_repos(self):
        records = _make_records({"A": 10, "B": 10, "C": 10})
        result = _stratified_sample(records, cap=9)
        assert len(result) == 9
        by_repo = {}
        for r in result:
            by_repo[r.repo] = by_repo.get(r.repo, 0) + 1
        # Each repo should get exactly 3
        assert by_repo.get("A", 0) == 3
        assert by_repo.get("B", 0) == 3
        assert by_repo.get("C", 0) == 3

    def test_cap_equals_total(self):
        records = _make_records({"A": 5, "B": 5})
        result = _stratified_sample(records, cap=10)
        assert len(result) == 10

    def test_heavily_imbalanced_repos(self):
        """Small repo should receive at least one slot even when it is much smaller."""
        # Big=8, Small=2, cap=5 → Big gets floor(5*8/10)=4, Small gets floor(5*2/10)=1
        # No remainder needed; Small is guaranteed 1 slot.
        records = _make_records({"Big": 8, "Small": 2})
        result = _stratified_sample(records, cap=5)
        assert len(result) == 5
        small_count = sum(1 for r in result if r.repo == "Small")
        assert small_count >= 1