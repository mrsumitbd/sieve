"""
tests/test_stratified_caps.py

Unit tests for _stratified_caps (module-level in pipeline.py).
"""

import random
import time

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


class TestLargestRemainderExact:
    @pytest.mark.parametrize("total_cap,expected", [
        (0, {"a": 0, "b": 0}),
        (1, {"a": 1, "b": 0}),
        (2, {"a": 1, "b": 1}),
        (3, {"a": 2, "b": 1}),
        (7, {"a": 4, "b": 3}),
        (15, {"a": 10, "b": 5}),
        (100, {"a": 10, "b": 5}),
    ])
    def test_two_repos(self, total_cap, expected):
        assert _stratified_caps({"a": 10, "b": 5}, total_cap) == expected

    def test_four_repos_remainders_split_by_count_then_name(self):
        counts = {"a": 100, "b": 50, "c": 25, "d": 25}
        assert _stratified_caps(counts, 10) == {"a": 4, "b": 2, "c": 2, "d": 2}

    def test_empty_repo_gets_zero(self):
        counts = {"a": 10, "b": 5, "c": 0}
        assert _stratified_caps(counts, 7) == {"a": 4, "b": 3, "c": 0}

    def test_cap_below_nonempty_repos_favours_largest(self):
        counts = {"a": 3, "b": 3, "c": 9, "d": 0}
        assert _stratified_caps(counts, 2) == {"a": 1, "b": 0, "c": 1, "d": 0}

    def test_cap_below_nonempty_repos_one_unit(self):
        counts = {"a": 3, "b": 8, "c": 5}
        assert _stratified_caps(counts, 1) == {"a": 0, "b": 1, "c": 0}

    def test_tie_goes_to_smaller_name(self):
        assert _stratified_caps({"a": 4, "b": 4}, 3) == {"a": 2, "b": 1}
        assert _stratified_caps({"b": 4, "a": 4}, 3) == {"a": 2, "b": 1}

    def test_equal_remainders_prefer_larger_count(self):
        # extra = 3, spare = (1, 2, 3), spare_total = 6: x and z both have
        # remainder 3 and one leftover unit remains. z wins on count even
        # though x sorts first by name.
        counts = {"x": 2, "y": 3, "z": 4}
        assert _stratified_caps(counts, 6) == {"x": 1, "y": 2, "z": 3}

    @pytest.mark.parametrize("total_cap,expected", [
        (0, 0),
        (3, 3),
        (7, 7),
        (50, 7),
    ])
    def test_single_repo(self, total_cap, expected):
        assert _stratified_caps({"x": 7}, total_cap) == {"x": expected}

    @pytest.mark.parametrize("bad", [-1, -5])
    def test_negative_cap_raises(self, bad):
        with pytest.raises(ValueError, match=str(bad)):
            _stratified_caps({"a": 10, "b": 5}, bad)

    def test_negative_cap_raises_on_empty_counts(self):
        with pytest.raises(ValueError):
            _stratified_caps({}, -1)


def _random_counts(rng: random.Random) -> dict[str, int]:
    n = rng.randint(0, 12)
    counts = {}
    for i in range(n):
        kind = rng.random()
        if kind < 0.25:
            counts[f"repo{i}"] = 0
        elif kind < 0.5:
            counts[f"repo{i}"] = rng.randint(1, 3)
        else:
            counts[f"repo{i}"] = rng.randint(1, 500)
    return counts


class TestLargestRemainderInvariants:
    def test_invariants_on_seeded_random_inputs(self):
        rng = random.Random(20240601)
        for _ in range(1000):
            counts = _random_counts(rng)
            available = sum(counts.values())
            total_cap = rng.choice([
                0, 1, rng.randint(0, 20), rng.randint(0, available + 50),
                available, available + 1,
            ])
            caps = _stratified_caps(counts, total_cap)
            effective = min(total_cap, available)
            nonempty = [r for r, c in counts.items() if c > 0]

            assert set(caps) == set(counts)
            assert sum(caps.values()) == effective
            for repo, cnt in counts.items():
                assert 0 <= caps[repo] <= cnt
                if cnt == 0:
                    assert caps[repo] == 0
            if effective >= len(nonempty):
                assert all(caps[r] >= 1 for r in nonempty)

            # deterministic and independent of insertion order
            assert _stratified_caps(counts, total_cap) == caps
            items = list(counts.items())
            rng.shuffle(items)
            assert _stratified_caps(dict(items), total_cap) == caps

    def test_proportionality_within_one_unit(self):
        rng = random.Random(7)
        for _ in range(300):
            counts = {f"r{i}": rng.randint(1, 200) for i in range(rng.randint(1, 10))}
            available = sum(counts.values())
            n = len(counts)
            total_cap = rng.randint(n, available)
            caps = _stratified_caps(counts, total_cap)
            spare_total = available - n
            extra = total_cap - n
            for repo, cnt in counts.items():
                if spare_total == 0:
                    continue
                exact = extra * (cnt - 1) / spare_total
                assert abs((caps[repo] - 1) - exact) < 1

    def test_large_input_is_fast(self):
        rng = random.Random(1)
        counts = {f"repo{i:05d}": rng.randint(0, 1000) for i in range(5000)}
        start = time.perf_counter()
        caps = _stratified_caps(counts, 100_000)
        elapsed = time.perf_counter() - start
        assert sum(caps.values()) == 100_000
        assert elapsed < 2.0

    def test_large_input_with_cap_below_repo_count(self):
        counts = {f"repo{i:05d}": i % 50 + 1 for i in range(5000)}
        caps = _stratified_caps(counts, 1234)
        assert sum(caps.values()) == 1234
        assert set(caps.values()) <= {0, 1}
