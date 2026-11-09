"""
tests/test_classifier.py

Unit tests for sieve.models.classifier.LLMCodeClassifier.

No GPU, network or real model weights are needed: loading is mocked, and
scoring runs on a tiny local model (tests/tiny_roberta.py). The batching,
truncation and device tests live in test_classifier_batching.py.
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

    @pytest.mark.parametrize("code", ["", "   ", "\n\t ", None])
    def test_score_returns_none_for_blank_input_without_scoring(self, tmp_path, code):
        clf = _make_classifier(tmp_path)
        clf._loaded = True
        # If the guard in score() were missing, this value would leak out.
        clf.score_batch = MagicMock(return_value=[0.9])
        assert clf.score(code) is None
        clf.score_batch.assert_not_called()

    def test_score_delegates_to_score_batch(self, tmp_path):
        clf = self._clf_with_mock_batch(tmp_path, [0.42])
        clf.score("def bar(): return 1")
        clf.score_batch.assert_called_once()


# ── score_batch ───────────────────────────────────────────────────────────────

# Only TestScoreBatch needs torch; a module-level importorskip would also skip
# the torch-free tests above and below it.
try:
    import torch
except ImportError:                                  # pragma: no cover
    torch = None

requires_torch = pytest.mark.skipif(
    torch is None, reason="torch not installed — skipping score_batch tests"
)


@requires_torch
class TestScoreBatch:
    """
    Behavioural contract of score_batch, exercised on the tiny local model
    (tests/tiny_roberta.py) instead of mocked tensors — the mocks hard-coded
    512-wide inputs, which dynamic padding deliberately no longer produces.
    """

    def _make_loaded_clf(self, tmp_path):
        pytest.importorskip("transformers", reason="transformers not installed")
        import tiny_roberta
        return tiny_roberta.make_classifier(tmp_path)

    def test_empty_list_returns_empty(self, tmp_path):
        clf = _make_classifier(tmp_path)
        clf._loaded = True
        result = clf.score_batch([])
        assert result == []

    def test_none_for_empty_snippets(self, tmp_path):
        clf = self._make_loaded_clf(tmp_path)
        results = clf.score_batch(["", "   ", None])
        assert results == [None, None, None]

    def test_length_matches_input(self, tmp_path):
        clf = self._make_loaded_clf(tmp_path)
        snippets = ["w1 w2", "w3 w4 w5", "w6"]
        results = clf.score_batch(snippets)
        assert len(results) == 3

    def test_scores_are_floats_in_unit_interval(self, tmp_path):
        clf = self._make_loaded_clf(tmp_path)
        results = clf.score_batch(["w1 w2 w3", "w9 w8"])
        assert all(isinstance(r, float) for r in results)
        assert all(0.0 <= r <= 1.0 for r in results)

    def test_mixed_empty_and_valid(self, tmp_path):
        clf = self._make_loaded_clf(tmp_path)
        results = clf.score_batch(["", "w1 w2 w3"])
        assert results[0] is None
        assert isinstance(results[1], float)

    def test_scores_are_rounded_to_4_decimal_places(self, tmp_path):
        clf = self._make_loaded_clf(tmp_path)
        results = clf.score_batch(["w1 w2 w3", "w4 w5"])
        assert all(r == round(r, 4) for r in results)

    def test_scores_are_sigmoid_of_the_model_logit(self, tmp_path):
        """The score is P = sigmoid(logit), not the raw logit."""
        clf = self._make_loaded_clf(tmp_path)
        enc = clf._tokenizer(["w1 w2 w3"], return_tensors="pt")
        with torch.no_grad():
            logit = clf._model(enc["input_ids"], enc["attention_mask"])
        assert clf.score_batch(["w1 w2 w3"])[0] == round(float(torch.sigmoid(logit)), 4)


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

    def test_llm_score_field_exists_on_function_record(self):
        func = self._make_function_record()
        assert hasattr(func, "llm_score")
        assert func.llm_score is None

    def test_llm_score_field_exists_on_class_record(self):
        cls = self._make_class_record()
        assert hasattr(cls, "llm_score")
        assert cls.llm_score is None


    def test_llm_score_truncated_defaults_to_none_on_function_record(self):
        func = self._make_function_record()
        assert hasattr(func, "llm_score_truncated")
        assert func.llm_score_truncated is None

    def test_llm_score_truncated_defaults_to_none_on_class_record(self):
        cls = self._make_class_record()
        assert hasattr(cls, "llm_score_truncated")
        assert cls.llm_score_truncated is None

    def test_llm_score_truncated_is_exported_next_to_llm_score(self):
        from dataclasses import asdict, fields
        from sieve.core.extraction import FunctionRecord, ClassRecord
        for rec_cls, rec in ((FunctionRecord, self._make_function_record()),
                             (ClassRecord, self._make_class_record())):
            names = [f.name for f in fields(rec_cls)]
            assert names.index("llm_score_truncated") == names.index("llm_score") + 1
            assert "llm_score_truncated" in asdict(rec)
