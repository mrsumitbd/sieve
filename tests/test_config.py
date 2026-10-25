"""
tests/test_config.py

Unit tests for SIEVEConfig validation and defaults.
"""

from datetime import date

import pytest
from pydantic import ValidationError

from sieve.config import SIEVEConfig


class TestDateRange:
    def test_valid_range(self):
        cfg = SIEVEConfig(
            language="Python",
            start_date=date(2024, 1, 1),
            end_date=date(2025, 1, 1),
        )
        assert cfg.start_date < cfg.end_date

    def test_same_day_is_valid(self):
        cfg = SIEVEConfig(
            language="Python",
            start_date=date(2024, 6, 1),
            end_date=date(2024, 6, 1),
        )
        assert cfg.start_date == cfg.end_date

    def test_inverted_range_raises(self):
        with pytest.raises(ValidationError):
            SIEVEConfig(
                language="Python",
                start_date=date(2025, 1, 1),
                end_date=date(2024, 1, 1),
            )


class TestDefaults:
    def test_end_date_required(self):
        from pydantic import ValidationError
        with pytest.raises(ValidationError):
            SIEVEConfig(language="Python", start_date=date(2024, 1, 1))

    def test_granularity_default(self):
        cfg = SIEVEConfig(language="Python", start_date=date(2024, 1, 1), end_date=date(2025, 1, 1))
        assert "function" in cfg.granularity
        assert "class" in cfg.granularity

    def test_dedup_off_by_default(self):
        cfg = SIEVEConfig(language="Python", start_date=date(2024, 1, 1), end_date=date(2025, 1, 1))
        assert cfg.deduplicate is False

    def test_caps_default_to_none(self):
        cfg = SIEVEConfig(language="Python", start_date=date(2024, 1, 1), end_date=date(2025, 1, 1))
        assert cfg.max_repos is None
        assert cfg.max_functions is None
        assert cfg.max_classes is None

    def test_engineered_only_off_by_default(self):
        cfg = SIEVEConfig(language="Python", start_date=date(2024, 1, 1), end_date=date(2025, 1, 1))
        assert cfg.engineered_only is False


class TestLanguages:
    @pytest.mark.parametrize("lang", ["Python", "Java", "JavaScript"])
    def test_supported_languages_accepted(self, lang):
        cfg = SIEVEConfig(language=lang, start_date=date(2024, 1, 1), end_date=date(2025, 1, 1))
        assert cfg.language == lang

    def test_unsupported_language_raises(self):
        with pytest.raises(ValidationError):
            SIEVEConfig(language="Rust", start_date=date(2024, 1, 1), end_date=date(2025, 1, 1))


class TestGranularity:
    def test_empty_granularity_raises(self):
        with pytest.raises(ValidationError):
            SIEVEConfig(
                language="Python",
                start_date=date(2024, 1, 1),
                end_date=date(2025, 1, 1),
                granularity=[],
            )


class TestExportFormats:
    @pytest.mark.parametrize("fmt", ["jsonl", "parquet", "both"])
    def test_all_formats_accepted(self, fmt):
        cfg = SIEVEConfig(
            language="Python",
            start_date=date(2024, 1, 1),
            end_date=date(2025, 1, 1),
            export_format=fmt,
        )
        assert cfg.export_format == fmt

class TestExtractionCaps:
    @pytest.mark.parametrize("field", ["max_functions", "max_classes"])
    @pytest.mark.parametrize("bad", [-1, -5])
    def test_negative_cap_raises(self, field, bad):
        with pytest.raises(ValidationError, match=field):
            SIEVEConfig(
                language="Python",
                start_date=date(2024, 1, 1),
                end_date=date(2025, 1, 1),
                **{field: bad},
            )

    @pytest.mark.parametrize("field", ["max_functions", "max_classes"])
    @pytest.mark.parametrize("value", [None, 1, 500])
    def test_valid_cap_accepted(self, field, value):
        cfg = SIEVEConfig(
            language="Python",
            start_date=date(2024, 1, 1),
            end_date=date(2025, 1, 1),
            **{field: value},
        )
        assert getattr(cfg, field) == value


class TestCapsRejectZero:
    @pytest.mark.parametrize("field", ["max_repos", "max_functions", "max_classes"])
    def test_zero_rejected_with_clear_message(self, field):
        with pytest.raises(ValidationError) as exc:
            SIEVEConfig(
                language="Python",
                start_date=date(2024, 1, 1),
                end_date=date(2025, 1, 1),
                **{field: 0},
            )
        msg = str(exc.value)
        assert field in msg
        assert "None (no cap) or an integer >= 1" in msg

    @pytest.mark.parametrize("field", ["max_repos", "max_functions", "max_classes"])
    def test_none_means_no_cap(self, field):
        cfg = SIEVEConfig(
            language="Python",
            start_date=date(2024, 1, 1),
            end_date=date(2025, 1, 1),
            **{field: None},
        )
        assert getattr(cfg, field) is None


class TestStrictKeys:
    def test_unknown_key_rejected_and_named(self):
        with pytest.raises(ValidationError, match="max_repoz"):
            SIEVEConfig(
                language="Python",
                start_date=date(2024, 1, 1),
                end_date=date(2025, 1, 1),
                max_repoz=5,
            )

    def test_no_pydantic_class_config_deprecation(self):
        import warnings

        with warnings.catch_warnings():
            warnings.simplefilter("error")
            import importlib

            import sieve.config

            importlib.reload(sieve.config)

    def test_enum_values_still_plain_strings(self):
        cfg = SIEVEConfig(language="Python", start_date=date(2024, 1, 1), end_date=date(2025, 1, 1))
        assert cfg.language == "Python"
        assert cfg.granularity == ["function", "class"]
        assert cfg.export_format == "jsonl"


class TestMaxRepos:
    @pytest.mark.parametrize("bad", [-1, -5])
    def test_negative_max_repos_raises(self, bad):
        with pytest.raises(ValidationError, match="max_repos"):
            SIEVEConfig(
                language="Python",
                start_date=date(2024, 1, 1),
                end_date=date(2025, 1, 1),
                max_repos=bad,
            )

    @pytest.mark.parametrize("value", [None, 1, 500])
    def test_valid_max_repos_accepted(self, value):
        cfg = SIEVEConfig(
            language="Python",
            start_date=date(2024, 1, 1),
            end_date=date(2025, 1, 1),
            max_repos=value,
        )
        assert cfg.max_repos == value
