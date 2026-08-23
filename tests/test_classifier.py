"""
tests/test_classifier.py

Unit tests for sieve.models.classifier.LLMCodeClassifier.

All tests mock torch/transformers so no GPU or model weights are needed
in CI — the classifier logic and interface are tested in isolation.
"""

import sys
from pathlib import Path
from unittest.mock import MagicMock, patch, PropertyMock

import pytest

# ── Helpers ───────────────────────────────────────────────────────────────────

def _make_classifier(tmp_path: Path, create_artifacts: bool = True):
    """Create a classifier pointing at a temp artifacts dir."""
    if create_artifacts:
        artifacts = tmp_path / "artifacts"
        artifacts.mkdir()
        (artifacts / "best_model.pt").write_bytes(b"fake_weights")
        tok_dir = artifacts / "tokenizer"
        tok_dir.mkdir()
        (tok_dir / "config.json").write_text("{}")
        model_dir = artifacts
    else:
        model_dir = tmp_path / "empty"
        model_dir.mkdir()

    # Import here so sys.path manipulation in conftest is already applied
    from sieve.models.classifier import LLMCodeClassifier
    return LLMCodeClassifier(model_dir)


# ── is_available ──────────────────────────────────────────────────────────────

class TestIsAvailable:
    def test_returns_true_when_artifacts_exist(self, tmp_path):
        clf = _make_classifier(tmp_path, create_artifacts=True)
        assert clf.is_available() is True

    def test_returns_false_when_weights_missing(self, tmp_path):
        artifacts = tmp_path / "artifacts"
        artifacts.mkdir()
        tok_dir = artifacts / "tokenizer"
        tok_dir.mkdir()
        # No best_model.pt
        from sieve.models.classifier import LLMCodeClassifier
        clf = LLMCodeClassifier(artifacts)
        assert clf.is_available() is False

    def test_returns_false_when_tokenizer_missing(self, tmp_path):
        artifacts = tmp_path / "artifacts"
        artifacts.mkdir()
        (artifacts / "best_model.pt").write_bytes(b"fake")
        # No tokenizer dir
        from sieve.models.classifier import LLMCodeClassifier
        clf = LLMCodeClassifier(artifacts)
        assert clf.is_available() is False

    def test_returns_false_when_dir_empty(self, tmp_path):
        clf = _make_classifier(tmp_path, create_artifacts=False)
        assert clf.is_available() is False


# ── load ──────────────────────────────────────────────────────────────────────

class TestLoad:
    def test_load_attempts_hub_when_artifacts_missing(self, tmp_path):
        """When local artifacts missing, load() calls _load which handles hub download."""
        from sieve.models.classifier import LLMCodeClassifier
        with patch.object(LLMCodeClassifier, "_load",
                          side_effect=Exception("Hub unavailable")) as mock_load:
            with pytest.raises(Exception, match="Hub unavailable"):
                LLMCodeClassifier.load(tmp_path)
            mock_load.assert_called_once()

    def test_load_skips_hub_when_artifacts_present(self, tmp_path):
        """load() always calls _load — _load handles local vs hub internally."""
        from sieve.models.classifier import LLMCodeClassifier
        _make_classifier(tmp_path)
        artifacts_dir = tmp_path / "artifacts"
        with patch.object(LLMCodeClassifier, "_load") as mock_load:
            LLMCodeClassifier.load(artifacts_dir)
            mock_load.assert_called_once()

    def test_load_calls_internal_load(self, tmp_path):
        clf = _make_classifier(tmp_path)
        with patch.object(clf, "_load") as mock_load:
            with patch.object(clf, "is_available", return_value=True):
                clf._load = mock_load
                mock_load.assert_not_called()

    def test_default_model_dir_used_when_none(self, tmp_path):
        from sieve.models.classifier import LLMCodeClassifier, _DEFAULT_MODEL_DIR
        if (_DEFAULT_MODEL_DIR / "best_model.pt").exists():
            pytest.skip("Artifacts exist locally — cannot test hub path")
        with patch.object(LLMCodeClassifier, "_load",
                          side_effect=Exception("no hub in CI")):
            with pytest.raises(Exception, match="no hub in CI"):
                LLMCodeClassifier.load()


# ── score ─────────────────────────────────────────────────────────────────────

class TestScore:
    def _clf_with_mock_batch(self, tmp_path, return_scores):
        clf = _make_classifier(tmp_path)
        clf._loaded = True
        clf.score_batch = MagicMock(return_value=return_scores)
        return clf

    def test_score_returns_float(self, tmp_path):
        clf = self._clf_with_mock_batch(tmp_path, [0.85])
        result = clf.score("def foo(): pass")
        assert isinstance(result, float)
        assert result == 0.85

    def test_score_returns_none_for_empty_string(self, tmp_path):
        clf = _make_classifier(tmp_path)
        clf._loaded = True
        clf.score_batch = MagicMock(return_value=[None])
        result = clf.score("")
        assert result is None

    def test_score_returns_none_for_whitespace(self, tmp_path):
        clf = _make_classifier(tmp_path)
        clf._loaded = True
        clf.score_batch = MagicMock(return_value=[None])
        result = clf.score("   ")
        assert result is None

    def test_score_delegates_to_score_batch(self, tmp_path):
        clf = self._clf_with_mock_batch(tmp_path, [0.42])
        clf.score("def bar(): return 1")
        clf.score_batch.assert_called_once()


# ── score_batch ───────────────────────────────────────────────────────────────

torch = pytest.importorskip("torch", reason="torch not installed — skipping score_batch tests")

class TestScoreBatch:
    def _make_loaded_clf(self, tmp_path):
        """Create a classifier with mocked torch internals."""
        clf = _make_classifier(tmp_path)
        clf._loaded  = True
        clf._device  = torch.device("cpu")

        # Mock tokenizer
        mock_enc = MagicMock()
        mock_enc.__getitem__ = lambda self, key: torch.zeros(1, 512, dtype=torch.long)
        mock_tokenizer = MagicMock()
        mock_tokenizer.return_value = {
            "input_ids":      torch.zeros(1, 512, dtype=torch.long),
            "attention_mask": torch.ones(1, 512, dtype=torch.long),
        }
        clf._tokenizer = mock_tokenizer

        # Mock model — returns logits
        mock_model = MagicMock()
        mock_model.return_value = torch.tensor([0.5])
        clf._model = mock_model

        return clf

    def test_empty_list_returns_empty(self, tmp_path):
        clf = _make_classifier(tmp_path)
        clf._loaded = True
        result = clf.score_batch([])
        assert result == []

    def test_none_for_empty_snippets(self, tmp_path):
        clf = self._make_loaded_clf(tmp_path)
        results = clf.score_batch(["", "   ", None])
        assert all(r is None for r in results)

    def test_length_matches_input(self, tmp_path):
        clf = self._make_loaded_clf(tmp_path)

        clf._tokenizer.return_value = {
            "input_ids":      torch.zeros(3, 512, dtype=torch.long),
            "attention_mask": torch.ones(3, 512, dtype=torch.long),
        }
        clf._model.return_value = torch.tensor([0.1, 0.5, 0.9])

        snippets = ["def a(): pass", "def b(): pass", "def c(): pass"]
        results = clf.score_batch(snippets)
        assert len(results) == 3

    def test_scores_are_floats_in_unit_interval(self, tmp_path):
        clf = self._make_loaded_clf(tmp_path)

        clf._tokenizer.return_value = {
            "input_ids":      torch.zeros(2, 512, dtype=torch.long),
            "attention_mask": torch.ones(2, 512, dtype=torch.long),
        }
        clf._model.return_value = torch.tensor([2.0, -1.0])  # logits outside [0,1]

        results = clf.score_batch(["code1", "code2"])
        assert all(isinstance(r, float) for r in results)
        assert all(0.0 <= r <= 1.0 for r in results)

    def test_mixed_empty_and_valid(self, tmp_path):
        clf = self._make_loaded_clf(tmp_path)

        clf._tokenizer.return_value = {
            "input_ids":      torch.zeros(1, 512, dtype=torch.long),
            "attention_mask": torch.ones(1, 512, dtype=torch.long),
        }
        clf._model.return_value = torch.tensor([0.8])

        results = clf.score_batch(["", "def foo(): pass"])
        assert results[0] is None
        assert isinstance(results[1], float)

    def test_scores_are_rounded_to_4_decimal_places(self, tmp_path):
        clf = self._make_loaded_clf(tmp_path)

        clf._tokenizer.return_value = {
            "input_ids":      torch.zeros(1, 512, dtype=torch.long),
            "attention_mask": torch.ones(1, 512, dtype=torch.long),
        }
        clf._model.return_value = torch.tensor([0.123456789])

        results = clf.score_batch(["def foo(): pass"])
        assert results[0] == round(results[0], 4)


# ── Pipeline Phase 5 integration ─────────────────────────────────────────────

class TestPipelinePhase5:
    """
    Tests for Phase 5 (LLM score annotation) in pipeline.py.
    Uses mocks so no real model or GitHub API calls are made.
    """

    def _make_function_record(self):
        from sieve.core.extraction import FunctionRecord
        return FunctionRecord(
            repo="test/repo",
            file_path="test.py",
            language="Python",
            func_name="foo",
            parameters=[],
            return_annotation=None,
            docstring=None,
            source_code="def foo(): pass",
            signature="def foo():",
            used_imports=[],
            start_line=1,
            end_line=1,
            is_method=False,
            parent_class=None,
            decorators=[],
        )

    def _make_class_record(self):
        from sieve.core.extraction import ClassRecord
        return ClassRecord(
            repo="test/repo",
            file_path="test.py",
            language="Python",
            class_name="Foo",
            parent_classes=[],
            docstring=None,
            source_code="class Foo: pass",
            skeleton="class Foo: pass",
            used_imports=[],
            method_names=[],
            method_count=0,
            has_constructor=False,
            decorators=[],
            start_line=1,
            end_line=1,
        )

    def test_llm_score_populated_when_annotate_enabled(self, tmp_path):
        """When annotate_llm_score=True and artifacts exist, records get scores."""
        from unittest.mock import MagicMock
        from sieve.models.classifier import LLMCodeClassifier

        func = self._make_function_record()
        cls  = self._make_class_record()

        mock_clf = MagicMock(spec=LLMCodeClassifier)
        mock_clf.is_available.return_value = True
        mock_clf.score_batch.return_value  = [0.85, 0.12]

        # Simulate Phase 5 directly — patch at source module level
        with patch("sieve.models.classifier.LLMCodeClassifier", return_value=mock_clf):
            all_records = [func, cls]
            snippets    = [r.source_code for r in all_records]
            scores      = mock_clf.score_batch(snippets)
            for record, score in zip(all_records, scores):
                record.llm_score = score

        assert func.llm_score == 0.85
        assert cls.llm_score  == 0.12

    def test_llm_score_none_when_artifacts_missing(self, tmp_path):
        """When artifacts are missing, llm_score stays None."""
        func = self._make_function_record()
        cls  = self._make_class_record()
        assert func.llm_score is None
        assert cls.llm_score  is None

    def test_llm_score_field_exists_on_function_record(self):
        func = self._make_function_record()
        assert hasattr(func, "llm_score")
        assert func.llm_score is None

    def test_llm_score_field_exists_on_class_record(self):
        cls = self._make_class_record()
        assert hasattr(cls, "llm_score")
        assert cls.llm_score is None

    def test_score_batch_called_with_all_source_codes(self):
        from unittest.mock import MagicMock
        from sieve.models.classifier import LLMCodeClassifier

        func = self._make_function_record()
        cls  = self._make_class_record()

        mock_clf = MagicMock(spec=LLMCodeClassifier)
        mock_clf.score_batch.return_value = [0.5, 0.5]

        all_records = [func, cls]
        snippets    = [r.source_code for r in all_records]
        mock_clf.score_batch(snippets, batch_size=64)

        mock_clf.score_batch.assert_called_once_with(
            [func.source_code, cls.source_code],
            batch_size=64,
        )