"""
tests/test_classifier_batching.py

Hermetic tests for LLMCodeClassifier's batching, padding, truncation reporting
and device selection. A tiny randomly initialised RoBERTa and a word-level
tokenizer stand in for CodeBERT (see tests/tiny_roberta.py), so nothing is
downloaded and no real weights are needed.

Skipped cleanly when torch / transformers / tokenizers are absent.
"""

import random
from pathlib import Path

import pytest

torch = pytest.importorskip("torch", reason="torch not installed — skipping classifier batching tests")
pytest.importorskip("transformers", reason="transformers not installed")
pytest.importorskip("tokenizers", reason="tokenizers not installed")

import tiny_roberta as tiny                                      # noqa: E402
from tiny_roberta import snippet_with_total_tokens as snip       # noqa: E402

import sieve.models.classifier as classifier_module              # noqa: E402
from sieve.models.classifier import LLMCodeClassifier            # noqa: E402

TOL = 1e-4 + 1e-9    # scores are rounded to 4 d.p., so a rounding flip is exactly 1e-4


# ── Fixtures / helpers ────────────────────────────────────────────────────────

@pytest.fixture
def clf(tmp_path):
    return tiny.make_classifier(tmp_path)


class ForwardSpy:
    """Records every model call: (input_ids, attention_mask, inference_mode_flag)."""

    def __init__(self, clf):
        self.calls = []
        clf._model.register_forward_pre_hook(self._hook, with_kwargs=True)

    def _hook(self, module, args, kwargs):
        ids, mask = args[0], args[1]
        self.calls.append((ids.clone(), mask.clone(), torch.is_inference_mode_enabled()))

    @property
    def widths(self):
        return [c[0].shape[1] for c in self.calls]

    @property
    def batch_sizes(self):
        return [c[0].shape[0] for c in self.calls]

    def rows(self):
        """Unpadded token-id rows seen across all calls, as tuples."""
        out = []
        for ids, mask, _ in self.calls:
            for row, m in zip(ids, mask):
                out.append(tuple(row[m.bool()].tolist()))
        return out


class CountingTokenizer:
    """Delegates to a real tokenizer; counts how many texts it is asked to encode."""

    def __init__(self, inner):
        self._inner = inner
        self.texts_seen = 0
        self.calls = 0

    def __call__(self, texts, *args, **kwargs):
        self.calls += 1
        self.texts_seen += len(texts) if isinstance(texts, (list, tuple)) else 1
        return self._inner(texts, *args, **kwargs)

    def __getattr__(self, name):
        return getattr(self._inner, name)


def _mixed_snippets(n=24, seed=1, longest=700):
    """Distinct snippets of varied length, some beyond the 512 limit."""
    rng = random.Random(seed)
    lengths = [rng.randint(3, 120) for _ in range(n - 3)] + [510, 520, longest]
    rng.shuffle(lengths)
    return [snip(total, seed=i) for i, total in enumerate(lengths)]


# ── Tiny-model fixture sanity (so the equivalence tests can actually fail) ────

class TestFixtureHasTeeth:
    def test_scores_vary_across_snippets(self, clf):
        scores = clf.score_batch(_mixed_snippets())
        assert max(scores) - min(scores) > 0.1

    def test_ignoring_the_attention_mask_would_be_detected(self, clf):
        """If padding were not masked, dynamic vs 512 padding would diverge."""
        tok = clf._tokenizer
        text = [snip(20)]
        short = tok(text, padding=True, return_tensors="pt")["input_ids"]
        long_ = tok(text, padding="max_length", max_length=512, return_tensors="pt")["input_ids"]
        with torch.no_grad():
            a = clf._model(short, torch.ones_like(short))
            b = clf._model(long_, torch.ones_like(long_))     # mask ignored: pads attended
        assert (a - b).abs().max() > 1e-2

    def test_production_head_has_same_state_dict_keys_as_tiny_head(self):
        build = classifier_module._build_classifier_module
        prod = build(tiny.make_encoder(seed=1))
        local = tiny.make_head_model(tiny.make_encoder(seed=2))
        assert set(prod.state_dict()) == set(local.state_dict())
        prod.load_state_dict(local.state_dict())           # shapes agree too

    def test_production_head_reads_only_the_cls_position(self):
        """Replace every non-[CLS] hidden state with garbage: logits must not move."""
        model = classifier_module._build_classifier_module(tiny.make_encoder(seed=3)).eval()
        tok = tiny.make_tokenizer()
        enc = tok([snip(40, seed=5), snip(25, seed=6)], padding=True, return_tensors="pt")

        with torch.no_grad():
            baseline = model(enc["input_ids"], enc["attention_mask"])

            real_encoder = model.encoder

            class _Garbled(torch.nn.Module):
                config = real_encoder.config

                def forward(self, input_ids, attention_mask):
                    out = real_encoder(input_ids=input_ids, attention_mask=attention_mask)
                    h = out.last_hidden_state.clone()
                    h[:, 1:, :] = torch.randn_like(h[:, 1:, :]) * 100.0
                    out.last_hidden_state = h
                    return out

            model.encoder = _Garbled()
            garbled = model(enc["input_ids"], enc["attention_mask"])
        assert torch.allclose(baseline, garbled)


# ── 1. Dynamic padding ────────────────────────────────────────────────────────

class TestDynamicPadding:
    def test_batch_is_padded_to_longest_snippet_not_512(self, clf):
        spy = ForwardSpy(clf)
        clf.score_batch([snip(10), snip(30), snip(20)])
        assert spy.widths == [30]

    def test_never_wider_than_512_even_for_long_snippets(self, clf):
        spy = ForwardSpy(clf)
        clf.score_batch([snip(900), snip(40)])
        assert spy.widths == [512]

    def test_scores_match_old_max_length_padding(self, clf):
        snippets = _mixed_snippets()
        new = clf.score_batch(snippets)
        old = tiny.old_scores(clf, snippets)
        diffs = [abs(a - b) for a, b in zip(new, old)]
        assert max(diffs) <= TOL, f"max |new-old| = {max(diffs)}"

    def test_raw_logits_match_before_rounding(self, clf):
        """Stricter than the 4-d.p. scores: float noise only, on the raw logits."""
        tok = clf._tokenizer
        texts = [snip(15, 1), snip(60, 2), snip(33, 3)]
        dyn = tok(texts, padding=True, truncation=True, max_length=512, return_tensors="pt")
        full = tok(texts, padding="max_length", truncation=True, max_length=512, return_tensors="pt")
        with torch.no_grad():
            a = clf._model(dyn["input_ids"], dyn["attention_mask"])
            b = clf._model(full["input_ids"], full["attention_mask"])
        assert torch.allclose(a, b, atol=1e-5)

    def test_batch_composition_does_not_change_a_score(self, clf):
        """Same snippet alone, in a tiny batch, and in a big mixed batch."""
        target = snip(37, seed=99)
        others = _mixed_snippets(12, seed=7)
        alone = clf.score_batch([target])[0]
        mixed = clf.score_batch(others + [target])[-1]
        small = clf.score_batch(others + [target], batch_size=2)[-1]
        assert abs(alone - mixed) <= TOL
        assert abs(alone - small) <= TOL


# ── 2. Length-sorted batching ─────────────────────────────────────────────────

class TestLengthSortedBatching:
    def test_order_helper_is_a_true_permutation(self):
        lengths = [5, 300, 5, 12, 512, 1, 300, 77]
        order = classifier_module._length_sorted_order(lengths)
        assert sorted(order) == list(range(len(lengths)))
        assert [lengths[i] for i in order] == sorted(lengths, reverse=True)

    def test_order_helper_is_deterministic_and_stable_on_ties(self):
        lengths = [7, 7, 7, 3, 3]
        assert classifier_module._length_sorted_order(lengths) == [0, 1, 2, 3, 4]

    def test_unsort_roundtrip_restores_input_order(self):
        rng = random.Random(0)
        lengths = [rng.randint(1, 512) for _ in range(200)]
        order = classifier_module._length_sorted_order(lengths)
        in_sorted_order = [f"item-{i}" for i in order]
        restored = [None] * len(lengths)
        for pos, original_index in enumerate(order):
            restored[original_index] = in_sorted_order[pos]
        assert restored == [f"item-{i}" for i in range(len(lengths))]

    def test_helper_handles_empty(self):
        assert classifier_module._length_sorted_order([]) == []

    def test_every_snippet_reaches_the_model_exactly_once(self, clf):
        snippets = _mixed_snippets(30, seed=3)
        spy = ForwardSpy(clf)
        clf.score_batch(snippets, batch_size=7)

        tok = clf._tokenizer
        expected = sorted(tuple(tok(s, truncation=True, max_length=512)["input_ids"])
                          for s in snippets)
        assert sorted(spy.rows()) == expected

    def test_batches_are_sorted_longest_first(self, clf):
        spy = ForwardSpy(clf)
        clf.score_batch(_mixed_snippets(30, seed=4), batch_size=7)
        assert spy.widths == sorted(spy.widths, reverse=True)
        assert spy.widths[0] > spy.widths[-1]      # not vacuous: widths really differ

    def test_each_batch_pads_only_to_its_own_longest(self, clf):
        spy = ForwardSpy(clf)
        clf.score_batch(_mixed_snippets(30, seed=5), batch_size=7)
        for ids, mask, _ in spy.calls:
            assert ids.shape[1] == int(mask.sum(dim=1).max())

    def test_padding_uses_pad_token_and_zero_mask(self, clf):
        spy = ForwardSpy(clf)
        clf.score_batch([snip(10), snip(25)])
        ids, mask, _ = spy.calls[0]
        assert (ids[mask == 0] == tiny.PAD_ID).all()
        assert mask.sum(dim=1).tolist() == [25, 10]

    def test_output_order_equals_input_order_for_mixed_lengths(self, clf):
        snippets = _mixed_snippets(24, seed=6)
        batched = clf.score_batch(snippets, batch_size=5)
        one_by_one = [clf.score_batch([s])[0] for s in snippets]
        assert len(batched) == len(snippets)
        for pos, (b, s) in enumerate(zip(batched, one_by_one)):
            assert abs(b - s) <= TOL, f"position {pos} holds another snippet's score"

    def test_output_order_survives_reversal(self, clf):
        snippets = _mixed_snippets(16, seed=8)
        fwd = clf.score_batch(snippets, batch_size=4)
        rev = clf.score_batch(snippets[::-1], batch_size=4)[::-1]
        assert all(abs(a - b) <= TOL for a, b in zip(fwd, rev))

    def test_none_positions_unchanged_for_empty_snippets(self, clf):
        snippets = ["", snip(30), "   ", None, snip(8), "\n\t", snip(500)]
        result = clf.score_batch(snippets, batch_size=2)
        assert len(result) == len(snippets)
        assert [r is None for r in result] == [True, False, True, True, False, True, False]
        valid = [s for s in snippets if s and str(s).strip()]
        assert all(abs(a - b) <= TOL for a, b in
                   zip([r for r in result if r is not None], clf.score_batch(valid)))

    def test_all_empty_never_calls_the_model(self, clf):
        spy = ForwardSpy(clf)
        assert clf.score_batch(["", None, "  "]) == [None, None, None]
        assert spy.calls == []

    @pytest.mark.parametrize("n,batch_size,expected", [
        (10, 4, [4, 4, 2]),
        (8, 4, [4, 4]),
        (3, 64, [3]),
        (5, 1, [1, 1, 1, 1, 1]),
    ])
    def test_batch_size_semantics_unchanged(self, clf, n, batch_size, expected):
        spy = ForwardSpy(clf)
        clf.score_batch([snip(5 + i, seed=i) for i in range(n)], batch_size=batch_size)
        assert spy.batch_sizes == expected

    def test_batch_size_counts_only_valid_snippets(self, clf):
        spy = ForwardSpy(clf)
        clf.score_batch(["", snip(9), None, snip(7), "  ", snip(5)], batch_size=2)
        assert spy.batch_sizes == [2, 1]

    def test_each_snippet_is_tokenized_once(self, clf):
        clf._tokenizer = CountingTokenizer(clf._tokenizer)
        snippets = _mixed_snippets(30, seed=9) + ["", None]
        clf.score_batch(snippets, batch_size=7)
        assert clf._tokenizer.texts_seen == 30       # not 60, and empties are not tokenized


# ── 3. inference_mode ─────────────────────────────────────────────────────────

class TestInferenceMode:
    def test_forward_runs_under_inference_mode(self, clf):
        spy = ForwardSpy(clf)
        clf.score_batch([snip(12), snip(20)])
        assert spy.calls and all(flag for _, _, flag in spy.calls)

    def test_model_stays_in_eval_mode(self, clf):
        clf.score_batch([snip(12)])
        assert clf._model.training is False


# ── 5. Truncation reporting ───────────────────────────────────────────────────

class TestTruncationFlag:
    @pytest.mark.parametrize("total,expected", [
        (3, False), (100, False),
        (511, False), (512, False),       # 512 ids incl. <s> and </s> fits exactly
        (513, True), (514, True), (2000, True),
    ])
    def test_flag_at_the_boundary(self, clf, total, expected):
        (res,) = clf.score_batch_detailed([snip(total)])
        assert res.truncated is expected
        assert 0.0 <= res.score <= 1.0

    def test_flags_are_per_snippet_and_in_input_order(self, clf):
        snippets = [snip(513, 1), snip(40, 2), "", snip(512, 3), snip(700, 4), snip(511, 5)]
        res = clf.score_batch_detailed(snippets, batch_size=2)
        assert [None if r is None else r.truncated for r in res] == \
               [True, False, None, False, True, False]

    def test_truncated_input_is_cut_exactly_like_the_tokenizer_does(self, clf):
        text = snip(700, seed=11)
        spy = ForwardSpy(clf)
        clf.score_batch([text])
        native = clf._tokenizer(text, truncation=True, max_length=512)["input_ids"]
        assert len(native) == 512
        assert list(spy.rows()[0]) == native         # same ids, same trailing </s>

    def test_512_and_513_token_snippets_both_reach_the_model_as_512(self, clf):
        spy = ForwardSpy(clf)
        clf.score_batch([snip(512, 1), snip(513, 2)], batch_size=1)
        assert spy.widths == [512, 512]

    def test_untruncated_snippet_is_not_altered(self, clf):
        text = snip(512, seed=12)
        spy = ForwardSpy(clf)
        clf.score_batch([text])
        assert list(spy.rows()[0]) == clf._tokenizer(text)["input_ids"]

    def test_score_batch_returns_the_same_floats_as_detailed(self, clf):
        snippets = _mixed_snippets(10, seed=13) + [""]
        plain = clf.score_batch(snippets)
        detailed = clf.score_batch_detailed(snippets)
        assert plain == [None if d is None else d.score for d in detailed]

    def test_result_is_a_score_truncated_pair(self, clf):
        (res,) = clf.score_batch_detailed([snip(20)])
        score, truncated = res
        assert isinstance(score, float) and isinstance(truncated, bool)

    def test_empty_input(self, clf):
        assert clf.score_batch_detailed([]) == []


# ── 4. Device selection ───────────────────────────────────────────────────────

class TestDeviceSelection:
    @staticmethod
    def _avail(monkeypatch, cuda, mps):
        monkeypatch.setattr(torch.cuda, "is_available", lambda: cuda)
        monkeypatch.setattr(torch.backends.mps, "is_available", lambda: mps)

    @pytest.mark.parametrize("cuda,mps,expected", [
        (True, True, "cuda"), (True, False, "cuda"),
        (False, True, "mps"), (False, False, "cpu"),
    ])
    def test_auto_detect_order_is_cuda_then_mps_then_cpu(self, monkeypatch, cuda, mps, expected):
        self._avail(monkeypatch, cuda, mps)
        assert classifier_module._resolve_device(None).type == expected

    def test_explicit_cpu_wins_over_available_accelerators(self, monkeypatch):
        self._avail(monkeypatch, True, True)
        assert classifier_module._resolve_device("cpu").type == "cpu"

    @pytest.mark.parametrize("name", ["CPU", " cpu ", "Cpu"])
    def test_device_name_is_normalised(self, monkeypatch, name):
        self._avail(monkeypatch, False, False)
        assert classifier_module._resolve_device(name).type == "cpu"

    def test_explicit_mps_and_cuda_when_available(self, monkeypatch):
        self._avail(monkeypatch, True, True)
        assert classifier_module._resolve_device("mps").type == "mps"
        assert classifier_module._resolve_device("cuda").type == "cuda"
        d = classifier_module._resolve_device("cuda:1")
        assert (d.type, d.index) == ("cuda", 1)

    def test_cuda_requested_but_unavailable_is_a_clear_error(self, monkeypatch):
        self._avail(monkeypatch, False, True)
        with pytest.raises(RuntimeError, match="cuda.*not available"):
            classifier_module._resolve_device("cuda")

    def test_mps_requested_but_unavailable_is_a_clear_error(self, monkeypatch):
        self._avail(monkeypatch, True, False)
        with pytest.raises(RuntimeError, match="mps.*not available"):
            classifier_module._resolve_device("mps")

    @pytest.mark.parametrize("bad", ["tpu", "gpu", "cuda:", "cuda:x", "cuda:-1", "cpu:0",
                                     "mps:0", "xla", "", "  ", "cuda 0"])
    def test_unknown_device_strings_are_rejected(self, bad):
        with pytest.raises(ValueError, match=r"Unknown device.*cpu.*cuda.*mps"):
            classifier_module._resolve_device(bad)

    def test_error_message_quotes_the_offending_value(self):
        with pytest.raises(ValueError, match="'tpu'"):
            classifier_module._resolve_device("tpu")

    def test_load_rejects_bad_device_before_doing_any_work(self, tmp_path):
        with pytest.MonkeyPatch.context() as mp:
            called = []
            mp.setattr(LLMCodeClassifier, "_load", lambda self: called.append(1))
            with pytest.raises(ValueError, match="Unknown device"):
                LLMCodeClassifier.load(tmp_path, device="tpu")
        assert called == []

    def test_load_passes_device_through(self, tmp_path):
        seen = {}
        with pytest.MonkeyPatch.context() as mp:
            mp.setattr(LLMCodeClassifier, "_load",
                       lambda self: seen.update(device=self._requested_device))
            LLMCodeClassifier.load(tmp_path, device="cpu")
            assert seen["device"] == "cpu"
            LLMCodeClassifier.load(tmp_path)
            assert seen["device"] is None

    def test_real_load_path_honours_override_and_scores_like_the_source_model(
            self, tmp_path, monkeypatch):
        """Drive the real _load() with a tiny encoder/tokenizer and a saved state dict."""
        import transformers

        source = tiny.make_classifier(tmp_path / "src", seed=21)
        model_dir = tmp_path / "artifacts"
        model_dir.mkdir()
        torch.save(source._model.state_dict(), model_dir / "best_model.pt")

        monkeypatch.setattr(transformers.AutoTokenizer, "from_pretrained",
                            classmethod(lambda cls, *a, **k: tiny.make_tokenizer()))
        monkeypatch.setattr(transformers.AutoModel, "from_pretrained",
                            classmethod(lambda cls, *a, **k: tiny.make_encoder(seed=99)))
        monkeypatch.setattr(torch.cuda, "is_available", lambda: True)   # would win if auto
        monkeypatch.setattr(torch.backends.mps, "is_available", lambda: True)

        loaded = LLMCodeClassifier.load(model_dir, device="cpu")

        assert loaded._device.type == "cpu"
        assert next(loaded._model.parameters()).device.type == "cpu"
        assert loaded._model.training is False
        snippets = _mixed_snippets(8, seed=22)
        got, want = loaded.score_batch(snippets), source.score_batch(snippets)
        assert all(abs(a - b) <= TOL for a, b in zip(got, want))
