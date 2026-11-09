"""
tests/test_wait_for_rate_limit.py

The opt-in "wait for rate limit" mode: instead of dying when GitHub's quota runs
out, a run sleeps until the limit resets and carries on.

No network and no real sleeping: PyGithub is mocked (or driven through a stub
requester so the real PaginatedList is exercised), and the waiter is given a
fake clock whose sleep() just advances time.

github_limits is imported as a module and attributes are looked up inside each
test, so every test fails on its own until the feature exists.
"""

import json
import logging
import math
import re
from datetime import date, datetime, timedelta, timezone
from itertools import count
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, PropertyMock, patch

import pytest
import typer.main
from github.GithubException import GithubException, RateLimitExceededException
from github.PaginatedList import PaginatedList
from typer.testing import CliRunner

from sieve.cli import app
from sieve.config import SIEVEConfig
from sieve.core import discovery
from sieve.core import github_limits as gl
from sieve.core.discovery import DiscoveryReport, RepoMetadata, count_contributors, discover_repos
from sieve.core.github_limits import GitHubQuotaError, QuotaInfo
from sieve.pipeline import run_pipeline

UTC = timezone.utc
T0 = datetime(2030, 1, 1, 13, 0, 0, tzinfo=UTC)
TOKEN = "ghp_SECRET_TOKEN_1234567890"
START, END = date(2024, 1, 1), date(2024, 12, 31)
HOME = str(Path(__file__).resolve().parents[1] / "src" / "sieve" / "ui" / "Home.py")


# ─── Fakes ────────────────────────────────────────────────────────────────────

class FakeClock:
    """A clock whose sleep() advances it, so a wait takes no real time."""

    def __init__(self, start=T0):
        self.t = start
        self.sleeps = []

    def now(self):
        return self.t

    def sleep(self, seconds):
        self.sleeps.append(seconds)
        self.t += timedelta(seconds=seconds)

    @property
    def slept(self):
        return sum(self.sleeps)


def primary(reset, message="API rate limit exceeded for user ID 1."):
    """What PyGithub raises when the hourly quota is used up (reset = when it refills)."""
    return RateLimitExceededException(
        403, {"message": message},
        {"X-RateLimit-Remaining": "0", "X-RateLimit-Reset": str(int(reset.timestamp()))},
    )


def secondary(retry_after=None):
    headers = {} if retry_after is None else {"Retry-After": str(retry_after)}
    return RateLimitExceededException(
        403,
        {"message": "You have exceeded a secondary rate limit. Please wait a few minutes before you try again."},
        headers,
    )


def too_many_requests(retry_after=None):
    headers = {} if retry_after is None else {"Retry-After": str(retry_after)}
    return GithubException(429, {"message": "Too Many Requests"}, headers)


def flaky(*outcomes):
    """A callable that works through outcomes: an exception is raised, anything else returned."""
    queue = list(outcomes)

    def fn(*args, **kwargs):
        fn.calls += 1
        outcome = queue.pop(0) if len(queue) > 1 else queue[0]
        if isinstance(outcome, BaseException):
            raise outcome
        return outcome

    fn.calls = 0
    return fn


def waiter(clock, enabled=True, messages=None):
    return gl.RateLimitWaiter(
        enabled=enabled,
        progress=messages.append if messages is not None else None,
        sleep=clock.sleep,
        now=clock.now,
    )


# ─── Real PaginatedList behind a scripted requester ───────────────────────────

class PageStub:
    """
    Stands in for PyGithub's Requester under a real PaginatedList. Serves
    `pages` pages of `per_page` items; the request numbers in `fail` (1-based,
    counting every request, retries included) raise the given exception once.
    """

    def __init__(self, pages, per_page=2, fail=None, make_item=None):
        self.per_page = per_page
        self.pages = pages
        self.fail = dict(fail or {})
        self.make_item = make_item or (lambda i: {"id": i})
        self.requested = []          # page number of every request, failures included

    def requestJsonAndCheck(self, verb, url, parameters=None, headers=None, input=None):
        match = re.search(r"page=(\d+)", url)
        page = int(match.group(1)) if match else 1
        self.requested.append(page)
        if len(self.requested) in self.fail:
            raise self.fail.pop(len(self.requested))
        first = (page - 1) * self.per_page + 1
        items = [self.make_item(i) for i in range(first, first + self.per_page)]
        link = {}
        if page < self.pages:
            link = {"link": f'<https://api.github.com/items?page={page + 1}>; rel="next"'}
        return link, items


def paginated(stub, content=None):
    return PaginatedList(content or (lambda requester, headers, attrs: attrs["id"]),
                         stub, "/items", {})


class FlakyIterable:
    """A re-iterable that is not a PyGithub list; raises `error` once, before yielding index `fail_at`."""

    def __init__(self, items, fail_at, error):
        self.items, self.fail_at, self.error = list(items), fail_at, error

    def __iter__(self):
        for index, item in enumerate(self.items):
            if index == self.fail_at and self.error is not None:
                error, self.error = self.error, None
                raise error
            yield item


# ─── Discovery fakes ──────────────────────────────────────────────────────────

def users(n, bots=0):
    return ([SimpleNamespace(login=f"user{i}", type="User") for i in range(n)]
            + [SimpleNamespace(login=f"app{i}[bot]", type="Bot") for i in range(bots)])


def mock_repo(name, n_users=5):
    repo = MagicMock()
    repo.full_name = name
    repo.html_url = f"https://github.com/{name}"
    repo.stargazers_count = 100
    repo.default_branch = "main"
    repo.license = None
    repo.topics = ["t"]
    repo.pushed_at = None
    commit = MagicMock()
    commit.commit.committer.date = datetime(2024, 5, 1, tzinfo=UTC)
    repo.get_commits.return_value = [commit]
    repo.get_contributors.return_value = users(n_users)
    return repo


def run_discovery(search_result, limiter=None, report=None, token=TOKEN, sleep=None, **kwargs):
    """Run discover_repos on mocked PyGithub. `sleep` (a MagicMock) replaces time.sleep so a test can inspect it."""
    kwargs.setdefault("max_repos", 50)
    kwargs.setdefault("min_contributors", 1)
    sleep = sleep if sleep is not None else MagicMock()
    with patch("sieve.core.discovery.Github") as mock_gh, patch("sieve.core.discovery.time.sleep", sleep):
        mock_gh.return_value.search_repositories.return_value = search_result
        mock_gh.return_value.requester.requestJsonAndCheck.return_value = (
            {}, {"total_count": 3, "incomplete_results": False})
        client = mock_gh.return_value
        if "configure" in kwargs:
            kwargs.pop("configure")(client)
        return list(discover_repos("Python", START, END, github_token=token, report=report,
                                   limiter=limiter, **kwargs)), client


# ═══ 1. Config, CLI, UI ═══════════════════════════════════════════════════════

def make_config(**kw):
    return SIEVEConfig(language="Python", start_date=START, end_date=END, **kw)


class TestConfigField:
    def test_defaults_to_false(self):
        assert make_config().wait_for_rate_limit is False

    def test_can_be_enabled(self):
        assert make_config(wait_for_rate_limit=True).wait_for_rate_limit is True

    def test_has_a_description(self):
        field = SIEVEConfig.model_fields["wait_for_rate_limit"]
        assert field.description and "rate limit" in field.description.lower()

    def test_is_part_of_the_config_snapshot(self):
        assert make_config().model_dump(mode="json")["wait_for_rate_limit"] is False
        assert make_config(wait_for_rate_limit=True).model_dump(mode="json")["wait_for_rate_limit"] is True


_RUN = ["run", "--language", "Python", "--start-date", "2024-01-01", "--end-date", "2024-12-31"]
_FAKE_SUMMARY = {
    "total_repos_discovered": 0, "total_repos_processed": 0, "total_repos_failed": 0,
    "total_functions": 0, "total_classes": 0, "failed_repos": [], "output_paths": {}, "repo_stats": [],
}
runner = CliRunner()


def cli_config(extra=()):
    with patch("sieve.cli.run_pipeline", return_value=_FAKE_SUMMARY) as mock:
        result = runner.invoke(app, _RUN + list(extra), env={"GITHUB_TOKEN": "", "HF_TOKEN": ""})
    assert result.exit_code == 0, result.output
    return mock.call_args[0][0]


class TestCliFlag:
    def test_default_is_off(self):
        assert cli_config().wait_for_rate_limit is False

    def test_flag_turns_it_on(self):
        assert cli_config(["--wait-for-rate-limit"]).wait_for_rate_limit is True

    def test_negative_flag_turns_it_off(self):
        assert cli_config(["--no-wait-for-rate-limit"]).wait_for_rate_limit is False

    def test_both_spellings_are_declared_with_help_text(self):
        run_cmd = typer.main.get_command(app).commands["run"]
        (param,) = [p for p in run_cmd.params if p.name == "wait_for_rate_limit"]
        assert "--wait-for-rate-limit" in param.opts
        assert "--no-wait-for-rate-limit" in param.secondary_opts
        assert param.default is False
        assert param.help and "rate limit" in param.help.lower()

    def test_run_help_lists_the_flag(self):
        # Rich cuts long option names with "..." on a narrow terminal, so fix the width.
        result = runner.invoke(app, ["run", "--help"], env={"COLUMNS": "250"})
        assert result.exit_code == 0
        plain = re.sub(r"\x1b\[[0-9;]*m", "", result.output)
        assert "--wait-for-rate-limit" in plain
        assert "--no-wait-for-rate-limit" in plain
        assert "rate limit is reached" in plain

    def test_config_file_value_is_used(self, tmp_path):
        f = tmp_path / "config.json"
        f.write_text(json.dumps({"language": "Python", "start_date": "2024-01-01",
                                 "end_date": "2024-12-31", "wait_for_rate_limit": True}))
        assert cli_config(["--config", str(f)]).wait_for_rate_limit is True


class TestUiNeverWaits:
    """The hosted Space must not hang for an hour: the UI has no switch and always passes False."""

    @staticmethod
    def _run_ui():
        from streamlit.testing.v1 import AppTest

        at = AppTest.from_file(HOME, default_timeout=30).run()
        captured = {}

        def fake(config, *a, **kw):
            captured["config"] = config
            raise RuntimeError("stop")

        with patch("sieve.pipeline.run_pipeline", side_effect=fake):
            at.sidebar.button[0].click().run()
        return at, captured.get("config")

    def test_config_built_by_the_ui_has_waiting_off(self):
        _, config = self._run_ui()
        assert config is not None
        assert config.wait_for_rate_limit is False

    def test_ui_passes_false_explicitly(self):
        source = Path(HOME).read_text()
        assert re.search(r"wait_for_rate_limit\s*=\s*False", source)

    def test_ui_offers_no_switch_for_it(self):
        from streamlit.testing.v1 import AppTest

        at = AppTest.from_file(HOME, default_timeout=30).run()
        labels = [w.label for kind in ("toggle", "checkbox", "radio", "selectbox")
                  for w in getattr(at.sidebar, kind)]
        assert labels
        assert not [l for l in labels if "wait" in l.lower() or "rate limit" in l.lower()]


# ═══ 2. Classification ════════════════════════════════════════════════════════

class TestClassify:
    def test_primary_limit_carries_its_reset_time(self):
        hit = gl.classify_rate_limit(primary(T0 + timedelta(minutes=30)))
        assert hit.kind == "primary"
        assert hit.reset == T0 + timedelta(minutes=30)

    def test_primary_limit_without_reset_header(self):
        hit = gl.classify_rate_limit(RateLimitExceededException(403, {"message": "API rate limit exceeded"}, {}))
        assert hit.kind == "primary" and hit.reset is None

    def test_plain_403_that_mentions_a_rate_limit_is_primary(self):
        exc = GithubException(403, {"message": "API rate limit exceeded for 1.2.3.4"}, {})
        assert gl.classify_rate_limit(exc).kind == "primary"

    def test_secondary_403_uses_retry_after(self):
        hit = gl.classify_rate_limit(secondary(42))
        assert hit.kind == "secondary" and hit.retry_after == 42

    def test_secondary_without_retry_after(self):
        hit = gl.classify_rate_limit(secondary())
        assert hit.kind == "secondary" and hit.retry_after is None

    def test_429_with_retry_after_is_secondary(self):
        hit = gl.classify_rate_limit(too_many_requests(7))
        assert hit.kind == "secondary" and hit.retry_after == 7

    def test_retry_after_header_name_is_case_insensitive(self):
        exc = GithubException(429, {"message": "slow down"}, {"retry-after": "9"})
        assert gl.classify_rate_limit(exc).retry_after == 9

    @pytest.mark.parametrize("exc", [
        GithubException(404, {"message": "Not Found"}, {}),
        GithubException(500, {"message": "Server Error"}, {}),
        GithubException(403, {"message": "The history or contributor list is too large to list contributors"}, {}),
        GithubException(401, {"message": "Bad credentials"}, {}),
        ValueError("rate limit"),
    ])
    def test_everything_else_is_not_a_rate_limit(self, exc):
        assert gl.classify_rate_limit(exc) is None


# ═══ 3. The waiter around one call ════════════════════════════════════════════

class TestPrimaryLimit:
    def test_waits_until_reset_plus_buffer_then_retries_the_same_call(self):
        clock = FakeClock()
        reset = T0 + timedelta(seconds=120)
        fn = flaky(primary(reset), "ok")
        w = waiter(clock)

        assert w.call(fn) == "ok"

        assert fn.calls == 2
        assert clock.slept == 120 + gl.WAIT_BUFFER_SECONDS
        assert clock.now() == reset + timedelta(seconds=gl.WAIT_BUFFER_SECONDS)
        assert 0 < gl.WAIT_BUFFER_SECONDS <= 30

    def test_sleeps_in_short_chunks(self):
        clock = FakeClock()
        waiter(clock).call(flaky(primary(T0 + timedelta(minutes=10)), "ok"))
        assert len(clock.sleeps) > 100
        assert all(0 < s <= gl.WAIT_CHUNK_SECONDS for s in clock.sleeps)
        assert gl.WAIT_CHUNK_SECONDS <= 10

    def test_reset_already_in_the_past_waits_only_the_buffer(self):
        clock = FakeClock()
        waiter(clock).call(flaky(primary(T0 - timedelta(seconds=30)), "ok"))
        assert clock.slept == gl.WAIT_BUFFER_SECONDS

    def test_unknown_reset_falls_back_to_a_fixed_wait(self):
        clock = FakeClock()
        exc = RateLimitExceededException(403, {"message": "API rate limit exceeded"}, {})
        waiter(clock).call(flaky(exc, "ok"))
        assert clock.slept == gl.UNKNOWN_RESET_WAIT_SECONDS

    def test_arguments_are_passed_through(self):
        seen = []
        w = waiter(FakeClock())
        assert w.call(lambda a, b=0: seen.append((a, b)) or "r", 1, b=2) == "r"
        assert seen == [(1, 2)]

    def test_each_wait_is_counted(self):
        clock = FakeClock()
        w = waiter(clock)
        w.call(flaky(primary(T0 + timedelta(seconds=60)), "a"))
        w.call(flaky(primary(clock.now() + timedelta(seconds=30)), "b"))
        assert w.stats() == {"count": 2, "seconds": 60 + 30 + 2 * gl.WAIT_BUFFER_SECONDS}

    def test_gives_up_when_the_limit_never_clears(self):
        clock = FakeClock()
        exc = primary(T0)
        fn = flaky(exc)
        with pytest.raises(RateLimitExceededException) as info:
            waiter(clock).call(fn)
        assert info.value is exc
        assert fn.calls == gl.MAX_WAITS_PER_CALL + 1


class TestSecondaryLimit:
    def test_waits_retry_after(self):
        clock = FakeClock()
        fn = flaky(secondary(42), "ok")
        assert waiter(clock).call(fn) == "ok"
        assert clock.slept == 42
        assert fn.calls == 2

    def test_defaults_to_sixty_seconds_without_retry_after(self):
        clock = FakeClock()
        waiter(clock).call(flaky(secondary(), "ok"))
        assert clock.slept == 60 == gl.SECONDARY_DEFAULT_WAIT_SECONDS

    def test_429_with_retry_after(self):
        clock = FakeClock()
        waiter(clock).call(flaky(too_many_requests(7), "ok"))
        assert clock.slept == 7

    def test_429_without_retry_after_uses_the_default(self):
        clock = FakeClock()
        waiter(clock).call(flaky(too_many_requests(), "ok"))
        assert clock.slept == gl.SECONDARY_DEFAULT_WAIT_SECONDS

    def test_retry_after_is_not_stretched_to_a_reset_time(self):
        clock = FakeClock()
        exc = secondary(5)
        exc.headers["X-RateLimit-Reset"] = str(int((T0 + timedelta(hours=1)).timestamp()))
        waiter(clock).call(flaky(exc, "ok"))
        assert clock.slept == 5


class TestWaitingDisabled:
    def test_the_original_exception_is_raised_untouched(self):
        clock = FakeClock()
        exc = primary(T0 + timedelta(hours=1))
        fn = flaky(exc)
        with pytest.raises(RateLimitExceededException) as info:
            waiter(clock, enabled=False).call(fn)
        assert info.value is exc
        assert fn.calls == 1
        assert clock.sleeps == []

    def test_a_secondary_limit_is_raised_untouched(self):
        exc = secondary(30)
        with pytest.raises(RateLimitExceededException) as info:
            waiter(FakeClock(), enabled=False).call(flaky(exc))
        assert info.value is exc

    def test_nothing_is_counted(self):
        w = waiter(FakeClock(), enabled=False)
        assert w.stats() == {"count": 0, "seconds": 0}

    def test_default_waiter_is_disabled(self):
        assert gl.RateLimitWaiter().enabled is False

    def test_enabled_waiter_does_not_swallow_other_errors(self):
        clock = FakeClock()
        exc = GithubException(500, {"message": "boom"}, {})
        fn = flaky(exc)
        with pytest.raises(GithubException) as info:
            waiter(clock).call(fn)
        assert info.value is exc
        assert fn.calls == 1 and clock.sleeps == []

    def test_a_rejected_token_is_never_waited_out(self):
        from github.GithubException import BadCredentialsException

        clock = FakeClock()
        with pytest.raises(BadCredentialsException):
            waiter(clock).call(flaky(BadCredentialsException(401, {"message": "Bad credentials"}, {})))
        assert clock.sleeps == []


class TestProgressMessages:
    def test_message_names_the_reset_time_and_the_remaining_wait(self):
        clock, messages = FakeClock(), []
        waiter(clock, messages=messages).call(flaky(primary(T0 + timedelta(seconds=120)), "ok"))

        first = messages[0]
        assert "rate limit" in first.lower()
        assert "13:02 UTC" in first                                  # when GitHub resets
        assert gl.format_duration(120 + gl.WAIT_BUFFER_SECONDS) in first   # how long we will wait

    def test_remaining_wait_counts_down_and_the_end_is_announced(self):
        clock, messages = FakeClock(), []
        waiter(clock, messages=messages).call(flaky(primary(T0 + timedelta(seconds=120)), "ok"))

        assert len(messages) == 4
        assert gl.format_duration(65) in messages[1]       # after one progress interval
        assert gl.format_duration(5) in messages[2]
        assert "resum" in messages[3].lower()
        assert gl.WAIT_PROGRESS_INTERVAL_SECONDS == 60

    def test_a_long_wait_does_not_flood_the_log(self):
        clock, messages = FakeClock(), []
        waiter(clock, messages=messages).call(flaky(primary(T0 + timedelta(hours=1)), "ok"))
        assert len(messages) < 70          # about one a minute, not one per chunk

    def test_secondary_message_names_the_pause(self):
        clock, messages = FakeClock(), []
        waiter(clock, messages=messages).call(flaky(secondary(42), "ok"))
        assert "secondary" in messages[0].lower()
        assert gl.format_duration(42) in messages[0]

    def test_without_a_callback_messages_go_to_the_log(self, caplog):
        with caplog.at_level(logging.INFO, logger="sieve.core.github_limits"):
            gl.RateLimitWaiter(enabled=True, sleep=FakeClock().sleep, now=FakeClock().now).call(
                flaky(secondary(3), "ok"))
        assert any("secondary" in r.getMessage().lower() for r in caplog.records)

    @pytest.mark.parametrize("seconds,text", [
        (0, "0s"), (0.4, "1s"), (5, "5s"), (59, "59s"), (60, "1m 00s"), (125, "2m 05s"),
        (3599, "59m 59s"), (3600, "1h 00m 00s"), (3725, "1h 02m 05s"),
    ])
    def test_format_duration(self, seconds, text):
        assert gl.format_duration(seconds) == text


class TestInterrupt:
    def test_ctrl_c_during_a_wait_propagates_and_the_call_is_not_retried(self):
        clock = FakeClock()
        sleeps = []

        def sleep(seconds):
            sleeps.append(seconds)
            if len(sleeps) == 3:
                raise KeyboardInterrupt

        fn = flaky(primary(T0 + timedelta(minutes=5)), "ok")
        w = gl.RateLimitWaiter(enabled=True, sleep=sleep, now=clock.now)

        with pytest.raises(KeyboardInterrupt) as info:
            w.call(fn)

        assert fn.calls == 1
        assert len(sleeps) == 3                            # stopped at once, no further chunks
        assert info.value.__context__ is None              # no "during handling of the above exception" noise
        assert 0 < w.stats()["seconds"] < 300              # only the time actually slept is counted

    def test_ctrl_c_while_paging_propagates(self):
        stub = PageStub(pages=3, fail={2: primary(T0 + timedelta(minutes=5))})

        def sleep(seconds):
            raise KeyboardInterrupt

        w = gl.RateLimitWaiter(enabled=True, sleep=sleep, now=FakeClock().now)
        got = []
        with pytest.raises(KeyboardInterrupt):
            for item in w.iterate(lambda: paginated(stub)):
                got.append(item)
        assert got == [1, 2]
        assert stub.requested == [1, 2]                    # nothing was requested after the interrupt


# ═══ 4. Pagination: resume without duplicates or gaps ═════════════════════════

class TestPaginationResume:
    def test_pygithub_re_iteration_would_repeat_cached_items(self):
        """Why iterate() slices: a naive retry of `for x in lst` re-yields everything already fetched."""
        stub = PageStub(pages=3, fail={2: primary(T0)})
        lst = paginated(stub)
        seen = []
        with pytest.raises(RateLimitExceededException):
            for item in lst:
                seen.append(item)
        seen.extend(lst)                                   # the naive retry
        assert seen == [1, 2, 1, 2, 3, 4, 5, 6]            # 1 and 2 twice

    def test_limit_hit_mid_pagination_gives_every_item_exactly_once(self):
        clock = FakeClock()
        stub = PageStub(pages=4, fail={3: primary(T0 + timedelta(seconds=90))})
        w = waiter(clock)

        got = list(w.iterate(lambda: paginated(stub)))

        assert got == list(range(1, 9))                    # no duplicates, nothing skipped, in order
        assert w.stats()["count"] == 1
        assert clock.slept == 90 + gl.WAIT_BUFFER_SECONDS

    def test_earlier_pages_are_not_fetched_again(self):
        stub = PageStub(pages=4, fail={3: primary(T0)})
        list(waiter(FakeClock()).iterate(lambda: paginated(stub)))
        assert stub.requested == [1, 2, 3, 3, 4]           # page 3 asked for twice, 1 and 2 once

    def test_limit_on_the_very_first_page(self):
        stub = PageStub(pages=2, fail={1: primary(T0)})
        got = list(waiter(FakeClock()).iterate(lambda: paginated(stub)))
        assert got == [1, 2, 3, 4]
        assert stub.requested == [1, 1, 2]

    def test_limit_on_the_last_page(self):
        stub = PageStub(pages=3, fail={3: primary(T0)})
        got = list(waiter(FakeClock()).iterate(lambda: paginated(stub)))
        assert got == [1, 2, 3, 4, 5, 6]

    def test_several_limits_in_one_pagination(self):
        clock = FakeClock()
        stub = PageStub(pages=5, fail={2: primary(T0), 4: secondary(20)})
        w = waiter(clock)
        got = list(w.iterate(lambda: paginated(stub)))
        assert got == list(range(1, 11))
        assert w.stats()["count"] == 2

    def test_secondary_limit_mid_pagination(self):
        clock = FakeClock()
        stub = PageStub(pages=3, fail={2: secondary(33)})
        got = list(waiter(clock).iterate(lambda: paginated(stub)))
        assert got == list(range(1, 7))
        assert clock.slept == 33

    def test_a_limit_raised_when_the_list_is_created(self):
        clock = FakeClock()
        stub = PageStub(pages=2)
        make = flaky(primary(T0 + timedelta(seconds=10)), paginated(stub))
        got = list(waiter(clock).iterate(make))
        assert got == [1, 2, 3, 4]
        assert make.calls == 2

    def test_plain_iterables_resume_by_skipping_what_was_consumed(self):
        clock = FakeClock()
        make = lambda: FlakyIterable(range(10), fail_at=6, error=primary(T0))
        assert list(waiter(clock).iterate(make)) == list(range(10))

    def test_pagination_is_lazy_so_consumers_can_stop_early(self):
        stub = PageStub(pages=10)
        it = waiter(FakeClock()).iterate(lambda: paginated(stub))
        assert [next(it), next(it), next(it)] == [1, 2, 3]
        assert stub.requested == [1, 2]

    def test_disabled_iteration_behaves_as_before(self):
        exc = primary(T0 + timedelta(hours=1))
        stub = PageStub(pages=3, fail={2: exc})
        got = []
        with pytest.raises(RateLimitExceededException) as info:
            for item in waiter(FakeClock(), enabled=False).iterate(lambda: paginated(stub)):
                got.append(item)
        assert info.value is exc
        assert got == [1, 2]

    def test_disabled_iteration_passes_a_plain_list_through(self):
        assert list(waiter(FakeClock(), enabled=False).iterate(lambda: [3, 1, 2])) == [3, 1, 2]


# ═══ 5. Wiring into discovery ═════════════════════════════════════════════════

def user_stub(n, per_page=100, bots=0, fail=None):
    """A real PaginatedList of n contributors (+ bots) behind a stub requester."""
    people = users(n, bots)
    stub = PageStub(pages=max(1, math.ceil(len(people) / per_page)), per_page=per_page, fail=fail,
                    make_item=lambda i: {"login": people[i - 1].login, "type": people[i - 1].type}
                    if i <= len(people) else None)
    # PaginatedList skips None items, so the short last page is padded harmlessly
    return paginated(stub, lambda requester, headers, attrs: SimpleNamespace(**attrs)), stub


class TestDiscoveryWaits:
    def test_contributor_paging_survives_a_limit_mid_list(self):
        clock = FakeClock()
        listing, stub = user_stub(5, per_page=2, bots=1, fail={2: primary(T0 + timedelta(seconds=40))})
        repo = mock_repo("o/r")
        repo.get_contributors.return_value = listing

        assert count_contributors(repo, TOKEN, limiter=waiter(clock)) == 5     # the bot is not counted
        assert stub.requested == [1, 2, 2, 3]
        assert clock.slept == 40 + gl.WAIT_BUFFER_SECONDS

    def test_contributor_count_matches_a_run_without_any_limit(self):
        calm, _ = user_stub(7, per_page=3)
        hit, _ = user_stub(7, per_page=3, fail={2: primary(T0)})
        repo_calm, repo_hit = mock_repo("o/a"), mock_repo("o/b")
        repo_calm.get_contributors.return_value, repo_hit.get_contributors.return_value = calm, hit
        assert (count_contributors(repo_hit, TOKEN, limiter=waiter(FakeClock()))
                == count_contributors(repo_calm, TOKEN) == 7)

    def test_a_contributor_limit_raised_by_the_call_itself(self):
        clock = FakeClock()
        repo = mock_repo("o/r")
        repo.get_contributors.side_effect = [primary(T0 + timedelta(seconds=10)), users(4)]
        assert count_contributors(repo, TOKEN, limiter=waiter(clock)) == 4
        assert repo.get_contributors.call_count == 2

    def test_contributor_cost_is_ceil_n_over_100_requests(self):
        listing, stub = user_stub(250)
        repo = mock_repo("o/r")
        repo.get_contributors.return_value = listing
        assert count_contributors(repo, TOKEN) == 250
        assert len(stub.requested) == gl.contributor_pages(250) == 3

    def test_search_paging_survives_a_limit_mid_results(self):
        clock = FakeClock()
        stub = PageStub(pages=3, per_page=2, fail={2: primary(T0 + timedelta(seconds=75))},
                        make_item=lambda i: {"id": i})
        results = paginated(stub, lambda requester, headers, attrs: mock_repo(f"o/r{attrs['id']}"))

        found, _ = run_discovery(results, limiter=waiter(clock))

        assert [m.full_name for m in found] == [f"o/r{i}" for i in range(1, 7)]
        assert stub.requested == [1, 2, 2, 3]
        assert clock.slept == 75 + gl.WAIT_BUFFER_SECONDS

    def test_search_limit_without_waiting_is_the_existing_quota_error(self):
        stub = PageStub(pages=3, per_page=2, fail={2: primary(T0)})
        results = paginated(stub, lambda requester, headers, attrs: mock_repo(f"o/r{attrs['id']}"))
        got = []
        with pytest.raises(GitHubQuotaError):
            with patch("sieve.core.discovery.Github") as mock_gh, patch("sieve.core.discovery.time.sleep"):
                mock_gh.return_value.search_repositories.return_value = results
                for meta in discover_repos("Python", START, END, max_repos=50, github_token=TOKEN):
                    got.append(meta.full_name)
        assert got == ["o/r1", "o/r2"]

    def test_result_count_requests_wait_too(self):
        clock = FakeClock()

        def configure(client):
            client.requester.requestJsonAndCheck.side_effect = [
                secondary(15), ({}, {"total_count": 2, "incomplete_results": False})]

        found, client = run_discovery([mock_repo("o/a"), mock_repo("o/b")], limiter=waiter(clock),
                                      max_repos=None, configure=configure)
        assert len(found) == 2
        assert client.requester.requestJsonAndCheck.call_count == 2
        assert clock.slept == 15

    def test_public_count_results_waits(self):
        clock = FakeClock()
        with patch("sieve.core.discovery.make_client") as make:
            make.return_value.requester.requestJsonAndCheck.side_effect = [
                primary(T0 + timedelta(seconds=30)), ({}, {"total_count": 9, "incomplete_results": False})]
            assert discovery.count_results("q", TOKEN, limiter=waiter(clock)) == 9
        assert clock.slept == 30 + gl.WAIT_BUFFER_SECONDS

    def test_last_commit_lookup_waits_and_is_retried(self):
        clock = FakeClock()
        repo = mock_repo("o/r")
        commit = MagicMock()
        commit.commit.committer.date = datetime(2024, 7, 7, tzinfo=UTC)
        repo.get_commits.side_effect = [primary(T0 + timedelta(seconds=20)), [commit]]

        (meta,), _ = run_discovery([repo], limiter=waiter(clock))

        assert repo.get_commits.call_count == 2
        assert meta.last_commit_date.startswith("2024-07-07")
        assert clock.slept == 20 + gl.WAIT_BUFFER_SECONDS

    def test_secondary_limit_in_the_commit_lookup_waits_instead_of_falling_back(self):
        clock = FakeClock()
        repo = mock_repo("o/r")
        commit = MagicMock()
        commit.commit.committer.date = datetime(2024, 7, 7, tzinfo=UTC)
        repo.get_commits.side_effect = [secondary(30), [commit]]
        (meta,), _ = run_discovery([repo], limiter=waiter(clock))
        assert meta.last_commit_date.startswith("2024-07-07")
        assert clock.slept == 30

    def test_topics_cost_no_request(self):
        repo = mock_repo("o/r")
        (meta,), _ = run_discovery([repo], limiter=waiter(FakeClock()))
        assert meta.topics == ["t"]
        repo.get_topics.assert_not_called()

    def test_full_cost_of_an_accepted_repo_matches_calls_for_repo(self):
        listing, stub = user_stub(250)
        repo = mock_repo("o/r")
        repo.get_contributors.return_value = listing
        run_discovery([repo])
        assert len(stub.requested) + repo.get_commits.call_count == gl.calls_for_repo(250) == 4

    def test_waits_are_recorded_in_the_report(self):
        clock = FakeClock()
        repo = mock_repo("o/r")
        repo.get_contributors.side_effect = [primary(T0 + timedelta(seconds=55)), users(3)]
        report = DiscoveryReport()
        run_discovery([repo], limiter=waiter(clock), report=report)
        assert report.rate_limit_waits == {"count": 1, "seconds": 55 + gl.WAIT_BUFFER_SECONDS}

    def test_report_defaults_to_no_waits(self):
        report = DiscoveryReport()
        run_discovery([mock_repo("o/r")], report=report)
        assert report.rate_limit_waits == {"count": 0, "seconds": 0}

    def test_without_a_limiter_the_limit_is_raised_as_before(self):
        repo = mock_repo("o/r")
        repo.get_contributors.side_effect = primary(T0)
        with pytest.raises(RateLimitExceededException):
            run_discovery([repo])

    def test_without_a_limiter_the_commit_lookup_error_is_raised_as_before(self):
        repo = mock_repo("o/r")
        repo.get_commits.side_effect = primary(T0)
        with pytest.raises(RateLimitExceededException):
            run_discovery([repo])

    def test_a_disabled_limiter_changes_nothing(self):
        clock = FakeClock()
        repo = mock_repo("o/r")
        repo.get_contributors.side_effect = primary(T0)
        with pytest.raises(RateLimitExceededException):
            run_discovery([repo], limiter=waiter(clock, enabled=False))
        assert clock.sleeps == []


# ═══ 6. Wiring into the quality phase ═════════════════════════════════════════

class Counter:
    """Stands in for a PaginatedList: .totalCount is the request, and may hit a limit first."""

    def __init__(self, *outcomes):
        self.outcomes = list(outcomes)
        self.reads = 0

    @property
    def totalCount(self):
        self.reads += 1
        outcome = self.outcomes.pop(0) if len(self.outcomes) > 1 else self.outcomes[0]
        if isinstance(outcome, BaseException):
            raise outcome
        return outcome


def collect(limiter=None, releases=None, pulls=None, issues=None):
    from sieve.core.quality import collect_metrics

    with patch("sieve.core.quality.Github") as mock_gh:
        repo = mock_gh.return_value.get_repo.return_value
        repo.get_releases.return_value = releases or Counter(4)
        repo.get_pulls.return_value = pulls or Counter(10)
        repo.get_issues.return_value = issues or Counter(25)
        kwargs = {} if limiter is None else {"limiter": limiter}
        metrics = collect_metrics("o/r", "MIT", 5, None, TOKEN, **kwargs)
        return metrics, mock_gh.return_value


class TestQualityPhaseWaits:
    def test_a_limit_on_any_request_waits_and_only_that_request_is_repeated(self):
        clock = FakeClock()
        pulls = Counter(primary(T0 + timedelta(seconds=45)), 10)
        releases = Counter(4)

        metrics, client = collect(waiter(clock), releases=releases, pulls=pulls)

        assert (metrics.release_count, metrics.pull_request_count, metrics.issue_count) == (4, 10, 15)
        assert client.get_repo.call_count == 1
        assert releases.reads == 1
        assert pulls.reads == 2
        assert clock.slept == 45 + gl.WAIT_BUFFER_SECONDS

    def test_limit_when_looking_up_the_repo(self):
        from sieve.core.quality import collect_metrics

        clock = FakeClock()
        with patch("sieve.core.quality.Github") as mock_gh:
            repo = mock_gh.return_value.get_repo.return_value
            repo.get_releases.return_value = Counter(1)
            repo.get_pulls.return_value = Counter(2)
            repo.get_issues.return_value = Counter(3)
            mock_gh.return_value.get_repo.side_effect = [secondary(12), repo]
            metrics = collect_metrics("o/r", "MIT", 5, None, TOKEN, limiter=waiter(clock))
        assert metrics.release_count == 1
        assert clock.slept == 12

    def test_without_waiting_the_limit_is_raised_as_before(self):
        with pytest.raises(RateLimitExceededException):
            collect(releases=Counter(primary(T0)))

    def test_a_disabled_limiter_raises_as_before(self):
        with pytest.raises(RateLimitExceededException):
            collect(waiter(FakeClock(), enabled=False), releases=Counter(primary(T0)))

    def test_other_github_errors_are_still_tolerated(self):
        metrics, _ = collect(waiter(FakeClock()), releases=Counter(GithubException(404, {"message": "x"}, {})))
        assert metrics.release_count == 0


# ═══ 7. Wiring into the pipeline ══════════════════════════════════════════════

def meta(name):
    return RepoMetadata(
        full_name=name, url=f"https://github.com/{name}", stars=100, contributors=5,
        last_commit_date="2024-01-01T00:00:00", default_branch="main", license_spdx="MIT",
        language="Python", collected_at="2024-06-01T00:00:00+00:00", topics=[],
    )


def pipeline_config(tmp_path, wait=False, token=None, engineered=False, max_repos=None):
    return SIEVEConfig(
        language="Python", start_date=START, end_date=END, engineered_only=engineered, max_repos=max_repos,
        wait_for_rate_limit=wait, output_dir=str(tmp_path / "out"), github_token=token,
    )


def quota(remaining, reset, limit=60):
    return QuotaInfo(core_remaining=remaining, core_limit=limit, core_reset=reset,
                     search_remaining=10, authenticated=False)


def manifest(tmp_path):
    return json.loads((tmp_path / "out" / "manifest.json").read_text())


def sleeps_of(mock_sleep):
    return sum(c.args[0] for c in mock_sleep.call_args_list)


def gen_of(items, seen=None, hook=None):
    def gen(**kwargs):
        if seen is not None:
            seen.update(kwargs)
        if hook:
            hook(kwargs)
        yield from items
    return gen


class TestQuotaAtStart:
    @pytest.mark.parametrize("remaining", [0, 9])
    def test_exhausted_quota_waits_instead_of_raising(self, tmp_path, remaining):
        reset = datetime.now(UTC) + timedelta(seconds=300)
        messages = []
        with patch("sieve.pipeline._fetch_quota", side_effect=[quota(remaining, reset), quota(60, reset + timedelta(hours=1))]) as fetch, \
             patch("sieve.pipeline.discover_repos", side_effect=gen_of([meta("a/one")])) as discover, \
             patch("sieve.pipeline._clone_repo", return_value=False), \
             patch("time.sleep") as sleep:
            summary = run_pipeline(pipeline_config(tmp_path, wait=True),
                                   progress_callback=lambda m, c, t: messages.append(m))

        assert fetch.call_count == 2
        discover.assert_called_once()
        assert sleeps_of(sleep) == pytest.approx(300 + gl.WAIT_BUFFER_SECONDS, abs=3)
        assert summary["total_repos_discovered"] == 1
        assert any("rate limit" in m.lower() and "UTC" in m for m in messages)
        assert any(m.startswith("GitHub API: 60/60") for m in messages)      # the refreshed quota is reported

    def test_without_waiting_an_exhausted_quota_still_raises(self, tmp_path):
        reset = datetime.now(UTC) + timedelta(seconds=300)
        with patch("sieve.pipeline._fetch_quota", return_value=quota(0, reset)), \
             patch("sieve.pipeline.discover_repos") as discover, \
             patch("time.sleep") as sleep:
            with pytest.raises(GitHubQuotaError):
                run_pipeline(pipeline_config(tmp_path, wait=False))
        discover.assert_not_called()
        sleep.assert_not_called()

    def test_a_quota_that_never_recovers_still_ends_in_the_quota_error(self, tmp_path):
        reset = datetime.now(UTC) + timedelta(seconds=30)
        with patch("sieve.pipeline._fetch_quota", return_value=quota(0, reset)) as fetch, \
             patch("sieve.pipeline.discover_repos") as discover, \
             patch("time.sleep"):
            with pytest.raises(GitHubQuotaError):
                run_pipeline(pipeline_config(tmp_path, wait=True))
        discover.assert_not_called()
        assert fetch.call_count == gl.MAX_WAITS_PER_CALL + 1

    def test_healthy_quota_does_not_wait(self, tmp_path):
        reset = datetime.now(UTC) + timedelta(hours=1)
        with patch("sieve.pipeline._fetch_quota", return_value=quota(50, reset)), \
             patch("sieve.pipeline.discover_repos", side_effect=gen_of([])), \
             patch("time.sleep") as sleep:
            run_pipeline(pipeline_config(tmp_path, wait=True))
        sleep.assert_not_called()

    def test_unreadable_quota_does_not_wait(self, tmp_path):
        with patch("sieve.pipeline._fetch_quota", return_value=None), \
             patch("sieve.pipeline.discover_repos", side_effect=gen_of([])), \
             patch("time.sleep") as sleep:
            run_pipeline(pipeline_config(tmp_path, wait=True))
        sleep.assert_not_called()


class TestPipelineWiring:
    def test_discovery_gets_a_limiter_matching_the_config(self, tmp_path):
        for wait in (False, True):
            seen = {}
            with patch("sieve.pipeline._fetch_quota", return_value=None), \
                 patch("sieve.pipeline.discover_repos", side_effect=gen_of([], seen)):
                run_pipeline(pipeline_config(tmp_path, wait=wait))
            assert seen["limiter"].enabled is wait

    def test_collect_metrics_gets_the_same_limiter(self, tmp_path):
        from sieve.core.quality import RepoQualityMetrics

        seen_discovery, seen_quality = {}, {}

        def fake_collect(**kwargs):
            seen_quality.update(kwargs)
            return RepoQualityMetrics(
                full_name=kwargs["repo_full_name"], license_spdx="MIT", contributor_count=5,
                release_count=3, pull_request_count=1, issue_count=1, loc=1, comment_lines=1,
                code_ratio=0.5, passes_stage1=True, passes_stage2=True, passes_stage3=True)

        with patch("sieve.pipeline._fetch_quota", return_value=None), \
             patch("sieve.pipeline.discover_repos", side_effect=gen_of([meta("a/one")], seen_discovery)), \
             patch("sieve.pipeline.collect_metrics", side_effect=fake_collect), \
             patch("sieve.pipeline.apply_filters", side_effect=lambda ms: ms), \
             patch("sieve.pipeline._clone_repo", return_value=False), \
             patch("time.sleep"):
            run_pipeline(pipeline_config(tmp_path, wait=True, engineered=True))

        assert seen_quality["limiter"] is seen_discovery["limiter"]
        assert seen_quality["limiter"].enabled is True

    def test_manifest_snapshots_the_setting(self, tmp_path):
        for wait in (False, True):
            with patch("sieve.pipeline._fetch_quota", return_value=None), \
                 patch("sieve.pipeline.discover_repos", side_effect=gen_of([])):
                run_pipeline(pipeline_config(tmp_path, wait=wait))
            assert manifest(tmp_path)["config"]["wait_for_rate_limit"] is wait

    def test_manifest_discovery_block_records_no_waits_by_default(self, tmp_path):
        with patch("sieve.pipeline._fetch_quota", return_value=None), \
             patch("sieve.pipeline.discover_repos", side_effect=gen_of([])):
            run_pipeline(pipeline_config(tmp_path))
        assert manifest(tmp_path)["discovery"]["rate_limit_waits"] == {"count": 0, "seconds": 0}

    def test_manifest_counts_waits_from_every_phase(self, tmp_path):
        """Quota check + discovery + quality phase all feed one total."""
        from sieve.core.quality import RepoQualityMetrics

        reset = lambda s: datetime.now(UTC) + timedelta(seconds=s)

        def in_discovery(kwargs):
            kwargs["limiter"].call(flaky(primary(reset(100)), None))

        def fake_collect(**kwargs):
            kwargs["limiter"].call(flaky(secondary(20), None))
            return RepoQualityMetrics(
                full_name=kwargs["repo_full_name"], license_spdx="MIT", contributor_count=5,
                release_count=3, pull_request_count=1, issue_count=1, loc=1, comment_lines=1,
                code_ratio=0.5, passes_stage1=True, passes_stage2=True, passes_stage3=True)

        with patch("sieve.pipeline._fetch_quota", side_effect=[quota(0, reset(200)), quota(60, reset(3600))]), \
             patch("sieve.pipeline.discover_repos", side_effect=gen_of([meta("a/one")], hook=in_discovery)), \
             patch("sieve.pipeline.collect_metrics", side_effect=fake_collect), \
             patch("sieve.pipeline.apply_filters", side_effect=lambda ms: ms), \
             patch("sieve.pipeline._clone_repo", return_value=False), \
             patch("time.sleep"):
            run_pipeline(pipeline_config(tmp_path, wait=True, engineered=True))

        waits = manifest(tmp_path)["discovery"]["rate_limit_waits"]
        assert waits["count"] == 3
        assert waits["seconds"] == pytest.approx((200 + 5) + (100 + 5) + 20, abs=5)

    def test_end_to_end_with_mocked_github(self, tmp_path):
        """Real discovery code, mocked PyGithub: a limit in the middle of a run is waited out."""
        reset = datetime.now(UTC) + timedelta(seconds=90)
        repos = [mock_repo("o/a"), mock_repo("o/b")]
        repos[1].get_contributors.side_effect = [primary(reset), users(6)]

        with patch("sieve.core.discovery.Github") as mock_gh, \
             patch("sieve.pipeline._fetch_quota", return_value=None), \
             patch("sieve.pipeline._clone_repo", return_value=False), \
             patch("time.sleep") as sleep:
            mock_gh.return_value.search_repositories.return_value = repos
            summary = run_pipeline(pipeline_config(tmp_path, wait=True, token=TOKEN, max_repos=10))

        assert summary["total_repos_discovered"] == 2
        assert summary["rate_limited"] is False
        assert sleeps_of(sleep) == pytest.approx(90 + gl.WAIT_BUFFER_SECONDS, abs=3)
        block = manifest(tmp_path)["discovery"]
        assert block["rate_limit_waits"]["count"] == 1
        assert block["complete"] is True

    def test_end_to_end_without_waiting_still_stops_at_the_limit(self, tmp_path):
        repos = [mock_repo("o/a"), mock_repo("o/b")]
        repos[1].get_contributors.side_effect = primary(datetime.now(UTC) + timedelta(seconds=90))

        with patch("sieve.core.discovery.Github") as mock_gh, \
             patch("sieve.pipeline._fetch_quota", return_value=None), \
             patch("sieve.pipeline._clone_repo", return_value=False), \
             patch("time.sleep"):
            mock_gh.return_value.search_repositories.return_value = repos
            summary = run_pipeline(pipeline_config(tmp_path, wait=False, max_repos=10))

        assert summary["rate_limited"] is True
        assert summary["total_repos_discovered"] == 1

    def test_ctrl_c_while_waiting_propagates_out_of_the_pipeline(self, tmp_path):
        reset = datetime.now(UTC) + timedelta(seconds=600)
        with patch("sieve.pipeline._fetch_quota", return_value=quota(0, reset)), \
             patch("sieve.pipeline.discover_repos") as discover, \
             patch("time.sleep", side_effect=KeyboardInterrupt):
            with pytest.raises(KeyboardInterrupt):
                run_pipeline(pipeline_config(tmp_path, wait=True))
        discover.assert_not_called()

    def test_ctrl_c_in_the_cli_aborts_cleanly(self):
        with patch("sieve.cli.run_pipeline", side_effect=KeyboardInterrupt):
            result = runner.invoke(app, _RUN + ["--wait-for-rate-limit"])
        assert result.exit_code != 0
        assert "Traceback" not in result.output
        assert isinstance(result.exception, SystemExit)


class TestTokenNeverLeaks:
    def test_no_message_log_line_or_manifest_contains_the_token(self, tmp_path, caplog):
        reset = lambda s: datetime.now(UTC) + timedelta(seconds=s)
        hostile = primary(reset(60), message=f"API rate limit exceeded for token {TOKEN}")
        repos = [mock_repo("o/a"), mock_repo("o/b")]
        repos[0].get_contributors.side_effect = [hostile, users(5)]
        repos[1].get_commits.side_effect = [secondary(5), repos[0].get_commits.return_value]
        messages = []

        with caplog.at_level(logging.DEBUG), \
             patch("sieve.core.discovery.Github") as mock_gh, \
             patch("sieve.pipeline._fetch_quota", side_effect=[quota(0, reset(30)), quota(60, reset(3600))]), \
             patch("sieve.pipeline._clone_repo", return_value=False), \
             patch("time.sleep"):
            mock_gh.return_value.search_repositories.return_value = repos
            run_pipeline(pipeline_config(tmp_path, wait=True, token=TOKEN, max_repos=10),
                         progress_callback=lambda m, c, t: messages.append(m))

        waiting = [m for m in messages if "rate limit" in m.lower()]
        assert len(waiting) >= 3                                   # the waits really were announced
        everything = "\n".join(messages + [r.getMessage() for r in caplog.records]
                               + [(tmp_path / "out" / "manifest.json").read_text()])
        assert TOKEN not in everything
        assert "SECRET_TOKEN" not in everything

    def test_waiter_messages_never_embed_exception_text(self):
        messages = []
        exc = primary(T0 + timedelta(seconds=10), message=f"leaked {TOKEN}")
        waiter(FakeClock(), messages=messages).call(flaky(exc, "ok"))
        assert messages and all(TOKEN not in m for m in messages)


# ═══ 8. Quota-aware pacing ════════════════════════════════════════════════════

class TestPacing:
    def pace(self, remaining, limit):
        with patch("sieve.core.github_limits.time.sleep") as sleep:
            slept = gl.pace_requests(SimpleNamespace(rate_limiting=(remaining, limit)))
        return sleep, slept

    @pytest.mark.parametrize("remaining,limit", [(5000, 5000), (4990, 5000), (2500, 5000), (501, 5000), (60, 60), (7, 60)])
    def test_plenty_of_quota_never_sleeps(self, remaining, limit):
        sleep, slept = self.pace(remaining, limit)
        sleep.assert_not_called()
        assert slept == 0

    @pytest.mark.parametrize("remaining,limit", [(500, 5000), (499, 5000), (100, 5000), (0, 5000), (6, 60), (1, 60), (1500, 15000)])
    def test_low_quota_sleeps_a_fixed_delay(self, remaining, limit):
        sleep, slept = self.pace(remaining, limit)
        sleep.assert_called_once_with(gl.PACING_DELAY_SECONDS)
        assert slept == gl.PACING_DELAY_SECONDS

    def test_threshold_is_ten_percent_of_the_window(self):
        assert gl.PACING_LOW_FRACTION == 0.10
        assert 0 < gl.PACING_DELAY_SECONDS <= 2

    @pytest.mark.parametrize("value", [(-1, -1), (0, 0), (5, 0), ("x", "y"), None, (None, None)])
    def test_unknown_quota_never_sleeps(self, value):
        with patch("sieve.core.github_limits.time.sleep") as sleep:
            gl.pace_requests(SimpleNamespace(rate_limiting=value))
        sleep.assert_not_called()

    def test_a_client_that_cannot_report_its_quota_never_sleeps(self):
        client = MagicMock()                                  # rate_limiting is a MagicMock: not a pair of numbers
        with patch("sieve.core.github_limits.time.sleep") as sleep:
            gl.pace_requests(client)
        sleep.assert_not_called()

    def test_a_failing_quota_read_never_breaks_the_run(self):
        class Broken:
            @property
            def rate_limiting(self):
                raise OSError("offline")

        with patch("sieve.core.github_limits.time.sleep") as sleep:
            assert gl.pace_requests(Broken()) == 0
        sleep.assert_not_called()


def with_quota(sequence):
    """Configure the mocked client so successive rate_limiting reads follow `sequence` (last repeats)."""
    reads = count()

    def configure(client):
        values = list(sequence)
        type(client).rate_limiting = PropertyMock(side_effect=lambda: values[min(next(reads), len(values) - 1)])
    return configure


class TestDiscoveryPacing:
    def repos(self, n=4):
        return [mock_repo(f"o/r{i}") for i in range(n)]

    def test_a_normal_run_has_no_fixed_sleep_at_all(self):
        sleep = MagicMock()
        found, _ = run_discovery(self.repos(), configure=with_quota([(4900, 5000)]), sleep=sleep)
        assert len(found) == 4
        sleep.assert_not_called()

    def test_default_mock_client_never_sleeps(self):
        sleep = MagicMock()
        run_discovery(self.repos(), sleep=sleep)
        sleep.assert_not_called()

    def test_low_quota_paces_every_repo(self):
        sleep = MagicMock()
        found, _ = run_discovery(self.repos(3), configure=with_quota([(100, 5000)]), max_repos=None, sleep=sleep)
        assert len(found) == 3
        assert [c.args[0] for c in sleep.call_args_list] == [gl.PACING_DELAY_SECONDS] * 3

    def test_pacing_starts_only_once_quota_runs_low(self):
        sleep = MagicMock()
        run_discovery(self.repos(4), configure=with_quota([(4000, 5000), (3000, 5000), (400, 5000), (300, 5000)]),
                      max_repos=None, sleep=sleep)
        assert sleep.call_count == 2

    def test_repos_rejected_by_the_contributor_filter_are_paced_too(self):
        repos = [mock_repo("o/a", n_users=1), mock_repo("o/b", n_users=1)]
        sleep = MagicMock()
        found, _ = run_discovery(repos, configure=with_quota([(50, 5000)]), min_contributors=3, sleep=sleep)
        assert found == []
        assert sleep.call_count == 2

    def test_pacing_is_not_applied_after_the_last_wanted_repo(self):
        sleep = MagicMock()
        run_discovery(self.repos(3), configure=with_quota([(50, 5000)]), max_repos=1, sleep=sleep)
        sleep.assert_not_called()

    def test_pacing_reads_the_clients_recorded_headers_without_extra_requests(self):
        _, client = run_discovery(self.repos(2), configure=with_quota([(4000, 5000)]), max_repos=None)
        # rate_limiting is PyGithub's record of the last response's headers; nothing else was asked of GitHub
        client.get_rate_limit.assert_not_called()


# ═══ 9. Planning figures ══════════════════════════════════════════════════════

class TestPlanningFigures:
    def test_contributor_pages(self):
        assert [gl.contributor_pages(n) for n in (0, 1, 100, 101, 200, 250, 2009)] == [1, 1, 1, 2, 2, 3, 21]

    def test_calls_for_a_repo_are_pages_plus_the_commit_lookup(self):
        assert [gl.calls_for_repo(n) for n in (0, 100, 101, 250)] == [2, 2, 3, 4]

    def test_page_size_matches_the_client(self):
        assert gl.CONTRIBUTORS_PER_PAGE == discovery.GITHUB_SEARCH_PAGE_SIZE == 100

    def test_calls_per_repo_is_built_from_its_parts(self):
        assert gl.CALLS_PER_REPO == (gl.LATEST_COMMIT_CALLS + gl.PLANNED_CONTRIBUTOR_PAGES
                                     + gl.PLANNED_REJECTED_PER_ACCEPTED) == 5

    def test_unauthenticated_estimate(self):
        assert gl.estimate_repos(60) == 12
        assert gl.estimate_repos(gl.UNAUTHENTICATED_LIMIT) == 12

    def test_estimate_edges(self):
        assert [gl.estimate_repos(n) for n in (-5, 0, 4, 5, 9, 10)] == [0, 0, 0, 1, 1, 2]

    def test_the_run_gate_is_unchanged(self):
        assert gl.MIN_CALLS_REQUIRED == 10

    def test_no_token_message_quotes_the_new_estimate(self):
        assert "roughly 12 repositories" in gl.build_exhausted_message(T0, authenticated=False)


# ═══ 10. Pacing against a real PyGithub client ════════════════════════════════

class _GitHubLookalike:
    """
    A loopback HTTP server answering the three kinds of request discovery makes
    (search, contributors, latest commit) with GitHub-style X-RateLimit-*
    headers, and recording when each request arrived. Lets the real PyGithub
    Requester fill in `rate_limiting`, which a MagicMock cannot do.
    """

    def __init__(self, remaining, limit=5000, repos=3):
        import threading
        from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

        outer = self
        self.paths, self.stamps = [], []
        self.remaining, self.limit, self.repos = remaining, limit, repos

        class Handler(BaseHTTPRequestHandler):
            protocol_version = "HTTP/1.1"

            def do_GET(self):
                import time as _t
                outer.paths.append(self.path)
                outer.stamps.append(_t.monotonic())
                base = f"http://127.0.0.1:{outer.port}"
                if self.path.startswith("/search"):
                    body = {"total_count": outer.repos, "incomplete_results": False, "items": [
                        {"id": i, "name": f"r{i}", "full_name": f"o/r{i}", "html_url": f"https://github.com/o/r{i}",
                         "url": f"{base}/repos/o/r{i}", "stargazers_count": 100 - i, "default_branch": "main",
                         "license": None, "topics": ["t"], "pushed_at": "2024-05-01T00:00:00Z",
                         "owner": {"login": "o", "type": "User"}} for i in range(outer.repos)]}
                elif "/contributors" in self.path:
                    body = [{"login": f"u{k}", "type": "User"} for k in range(5)]
                elif "/commits" in self.path:
                    when = {"date": "2024-05-01T00:00:00Z", "name": "n", "email": "e@e"}
                    body = [{"sha": "a", "url": "u", "html_url": "u", "commit": {
                        "committer": when, "author": when, "message": "m", "url": "u",
                        "tree": {"sha": "t", "url": "u"}, "comment_count": 0}}]
                else:
                    body = {}
                    outer.paths.append("UNEXPECTED " + self.path)
                payload = json.dumps(body).encode()
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(payload)))
                self.send_header("X-RateLimit-Limit", str(outer.limit))
                self.send_header("X-RateLimit-Remaining", str(outer.remaining))
                self.send_header("X-RateLimit-Reset", "4102444800")
                self.end_headers()
                self.wfile.write(payload)

            def log_message(self, *args):
                pass

        self.httpd = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        self.port = self.httpd.server_port
        self.thread = threading.Thread(target=self.httpd.serve_forever, daemon=True)

    def __enter__(self):
        self.thread.start()
        return self

    def __exit__(self, *exc):
        self.httpd.shutdown()
        self.httpd.server_close()


def discover_against(server, real_sleep=False, **client_kwargs):
    """
    discover_repos with its real PyGithub client pointed at the loopback server.
    Returns (repos, sleeps); time.sleep is replaced (and its calls recorded) unless real_sleep.
    """
    real_make = discovery.make_client
    sleeps = []
    patches = [patch("sieve.core.discovery.make_client",
                     lambda token=None, **kw: real_make(token, base_url=f"http://127.0.0.1:{server.port}",
                                                        **{**client_kwargs, **kw}))]
    if not real_sleep:
        patches.append(patch("time.sleep", side_effect=sleeps.append))
    for p in patches:
        p.start()
    try:
        # A cap above the number of repositories, so the run ends with the search, not on the cap
        found = list(discover_repos("Python", START, END, max_repos=server.repos + 5, min_contributors=1,
                                    github_token=TOKEN))
    finally:
        for p in patches:
            p.stop()
    return found, sleeps


class TestPacingWithRealClient:
    def test_plenty_of_quota_means_no_pause_and_no_extra_request(self):
        with _GitHubLookalike(remaining=4900) as server:
            found, sleeps = discover_against(server, seconds_between_requests=0)
        assert len(found) == 3
        assert sleeps == []
        assert not [p for p in server.paths if "rate_limit" in p or p.startswith("UNEXPECTED")]
        assert len(server.paths) == 1 + 3 * 2               # one search page; contributors + latest commit each

    def test_low_quota_pauses_after_every_repo_using_the_response_headers(self):
        with _GitHubLookalike(remaining=200) as server:       # 4% of 5000
            found, sleeps = discover_against(server, seconds_between_requests=0)
        assert len(found) == 3
        assert sleeps == [gl.PACING_DELAY_SECONDS] * 3
        assert not [p for p in server.paths if "rate_limit" in p]

    def test_the_client_itself_keeps_requests_a_quarter_second_apart(self):
        """
        What bounds the request rate now that discovery has no fixed pause: PyGithub spaces
        consecutive requests by seconds_between_requests (0.25 s by default, in every version SIEVE
        supports), so at most 4 requests a second, 240 a minute, against GitHub's 900 points a minute.
        make_client must not switch that off.
        """
        with _GitHubLookalike(remaining=4900, repos=2) as server:
            discover_against(server, real_sleep=True)         # default client settings, real sleeping
        gaps = [b - a for a, b in zip(server.stamps, server.stamps[1:])]
        assert len(gaps) >= 4
        assert min(gaps) >= 0.24
