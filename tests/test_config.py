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

    def test_min_last_activity_defaults_to_none(self):
        cfg = SIEVEConfig(language="Python", start_date=date(2024, 1, 1), end_date=date(2025, 1, 1))
        assert cfg.min_last_activity is None

    def test_min_last_activity_can_be_set(self):
        cfg = SIEVEConfig(
            language="Python",
            start_date=date(2024, 1, 1),
            end_date=date(2025, 1, 1),
            min_last_activity=date(2025, 6, 1),
        )
        assert cfg.min_last_activity == date(2025, 6, 1)

    def test_min_last_activity_cannot_be_before_end_date(self):
        from pydantic import ValidationError
        with pytest.raises(ValidationError):
            SIEVEConfig(
                language="Python",
                start_date=date(2024, 1, 1),
                end_date=date(2025, 1, 1),
                min_last_activity=date(2024, 6, 1),
            )

    def test_granularity_default(self):
        cfg = SIEVEConfig(language="Python", start_date=date(2024, 1, 1), end_date=date(2025, 1, 1))
        assert "function" in cfg.granularity
        assert "class" in cfg.granularity

    def test_dedup_on_by_default(self):
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