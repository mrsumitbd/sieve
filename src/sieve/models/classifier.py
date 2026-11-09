"""
sieve/models/classifier.py

Wraps the fine-tuned CodeBERT classifier for use inside the SIEVE pipeline.
Scores each extracted code snippet with P(AI-generated) → llm_score field.

Model weights are hosted on HuggingFace Hub and downloaded automatically
on first use, then cached locally.

Usage:
    from sieve.models.classifier import LLMCodeClassifier

    clf = LLMCodeClassifier.load()  # downloads from HF Hub if needed
    scores = clf.score_batch(["def foo(): ...", "public void bar() {}"])

    # Force a device instead of auto-detecting (cuda, then mps, then cpu):
    clf = LLMCodeClassifier.load(device="cpu")

    # Scores plus whether each snippet was cut to the model's 512-token window:
    results = clf.score_batch_detailed(["def foo(): ..."])   # [ScoreResult(score, truncated)]

Inputs longer than MAX_TOKENS (512, counting the <s> and </s> special tokens)
are scored on their first 512 tokens only; ScoreResult.truncated reports that.
"""

import logging
import re
from pathlib import Path
from typing import NamedTuple, Optional, Sequence, Union

from pydantic import SecretStr

from sieve.config import reveal_secret

logger = logging.getLogger(__name__)

# HuggingFace Hub repo containing model weights
HF_REPO_ID = "mrahman2025/sieve-llm-classifier"

# Default local cache location
_DEFAULT_MODEL_DIR = Path(__file__).parent / "artifacts"

# CodeBERT's context window, special tokens included (<s> + 510 code tokens + </s>).
MAX_TOKENS = 512

# Snippets are tokenized this many at a time, so the transient Python lists stay
# small; the ids are then kept as compact int32 arrays until they are batched.
_TOKENIZE_CHUNK = 1024

_DEVICE_PATTERN = re.compile(r"^(cpu|mps|cuda(:\d+)?)$")


class ScoreResult(NamedTuple):
    """A scored snippet: P(AI-generated), and whether only its first 512 tokens were seen."""
    score: float
    truncated: bool


def _resolve_device(device: Optional[str] = None):
    """
    Map a device request to a torch.device.

    None auto-detects (cuda, then mps, then cpu). Otherwise `device` must be
    "cpu", "mps", "cuda" or "cuda:N" (case-insensitive); anything else raises
    ValueError, and asking for an accelerator this machine lacks raises
    RuntimeError.
    """
    import torch

    if device is None:
        if torch.cuda.is_available():
            return torch.device("cuda")
        if torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")

    name = _validated_device_name(device)
    if name.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError(f"device {device!r} requested but CUDA is not available on this machine")
    if name == "mps" and not torch.backends.mps.is_available():
        raise RuntimeError(f"device {device!r} requested but MPS is not available on this machine")
    return torch.device(name)


def _validated_device_name(device) -> str:
    """Normalise a device string, rejecting unknown ones. Needs no torch."""
    name = str(device).strip().lower()
    if not _DEVICE_PATTERN.match(name):
        raise ValueError(
            f"Unknown device {device!r}. Use 'cpu', 'cuda' (or 'cuda:N'), 'mps', "
            f"or None to auto-detect (cuda, then mps, then cpu)."
        )
    return name


def _length_sorted_order(lengths: Sequence[int]) -> list[int]:
    """
    Indices that visit `lengths` longest-first (ties keep input order).

    Batching consecutive items of this order keeps each batch's lengths close,
    so padding to the batch maximum wastes little; longest-first also surfaces
    an out-of-memory error on the first batch rather than after hours of work.
    The result is a permutation of range(len(lengths)); writing result i into
    slot order[i] restores the input order.
    """
    return sorted(range(len(lengths)), key=lambda i: -lengths[i])


def _build_classifier_module(encoder):
    """
    The classification head on top of a RoBERTa-style encoder.

    The head reads only the [CLS] (position 0) hidden state:
        encoder -> last_hidden_state[:, 0, :] -> dropout -> linear(hidden, 1)
    Padding therefore influences the score only through attention, which the
    attention_mask removes, so padding to the longest snippet in a batch gives
    the same scores as padding every snippet to 512.

    Attribute names (`encoder`, `dropout`, `classifier`) fix the state_dict keys
    of the released weights; do not rename them.
    """
    import torch.nn as nn

    class _CodeBERTClassifier(nn.Module):
        def __init__(self):
            super().__init__()
            self.encoder    = encoder
            self.dropout    = nn.Dropout(0.1)
            self.classifier = nn.Linear(encoder.config.hidden_size, 1)

        def forward(self, input_ids, attention_mask):
            out = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
            cls = out.last_hidden_state[:, 0, :]
            cls = self.dropout(cls)
            return self.classifier(cls).squeeze(-1)

    return _CodeBERTClassifier()


class LLMCodeClassifier:
    """
    Binary classifier: P(code is AI-generated).

    Wraps microsoft/codebert-base fine-tuned on 114K human/AI code pairs
    across Python, Java, JavaScript, and C++.

    On first use, model weights are automatically downloaded from HuggingFace
    Hub and cached locally.
    """

    def __init__(self, model_dir: Union[str, Path]):
        self.model_dir  = Path(model_dir)
        self._model     = None
        self._tokenizer = None
        self._device    = None
        self._requested_device: Optional[str] = None
        self._loaded    = False

    # ── Loading ───────────────────────────────────────────────────────────────

    @classmethod
    def load(
        cls,
        model_dir: Optional[Union[str, Path]] = None,
        hf_token: Union[str, SecretStr, None] = None,
        device: Optional[str] = None,
    ) -> "LLMCodeClassifier":
        """
        Load classifier. Uses local artifacts if best_model.pt exists in
        model_dir (default: the bundled artifacts directory); otherwise
        downloads the weights from HuggingFace Hub (cached after first use).

        Args:
            model_dir: Optional local path to model artifacts directory.
            hf_token:  Optional HuggingFace token for private repos.
            device:    Optional device override: "cpu", "cuda" (or "cuda:N"), or
                       "mps". Default None auto-detects (cuda, then mps, then cpu).
                       An unknown string raises ValueError immediately, before
                       anything is loaded or downloaded.
        """
        requested = None if device is None else _validated_device_name(device)
        instance = cls(model_dir or _DEFAULT_MODEL_DIR)
        instance._hf_token = hf_token
        instance._requested_device = requested
        instance._load()
        return instance

    def _load(self):
        """Load model and tokenizer — from HF Hub cache or local artifacts."""
        import torch
        from transformers import AutoModel, AutoTokenizer
        from huggingface_hub import hf_hub_download

        # Device selection: explicit override, else cuda → mps → cpu
        self._device = _resolve_device(self._requested_device)
        logger.info(f"LLMCodeClassifier: using device {self._device}")

        # ── Tokenizer: always load from microsoft/codebert-base ───────────────
        logger.info("Loading tokenizer from microsoft/codebert-base...")
        self._tokenizer = AutoTokenizer.from_pretrained(
            "microsoft/codebert-base",
            token=reveal_secret(getattr(self, "_hf_token", None)),
        )

        # ── Model weights: local artifacts first, else HF Hub ─────────────────
        local_weights = self.model_dir / "best_model.pt"
        if local_weights.exists():
            logger.info("Loading weights from local artifacts...")
            weights_path = local_weights
        else:
            logger.info(f"Downloading weights from HF Hub ({HF_REPO_ID})...")
            weights_path = hf_hub_download(
                repo_id=HF_REPO_ID,
                filename="best_model.pt",
                token=reveal_secret(getattr(self, "_hf_token", None)),
            )

        # ── Model architecture ────────────────────────────────────────────────
        encoder = AutoModel.from_pretrained("microsoft/codebert-base")
        self._model = _build_classifier_module(encoder).to(self._device)
        state = torch.load(weights_path, map_location=self._device)
        self._model.load_state_dict(state)
        self._model.eval()
        self._loaded = True
        logger.info("LLMCodeClassifier: model loaded and ready")

    # ── Scoring ───────────────────────────────────────────────────────────────

    def score(self, code: str) -> Optional[float]:
        """
        Score a single code snippet.
        Returns P(AI-generated) ∈ [0, 1], or None if code is empty.
        """
        if not code or not str(code).strip():
            return None
        results = self.score_batch([code])
        return results[0]

    def score_batch(
        self,
        snippets: list[str],
        batch_size: int = 64,
    ) -> list[Optional[float]]:
        """
        Score a list of code snippets.
        Returns list of P(AI-generated) ∈ [0, 1] (None for empty snippets), in
        input order. Use score_batch_detailed to also learn which snippets were
        truncated to the model's 512-token window.
        """
        return [
            None if r is None else r.score
            for r in self.score_batch_detailed(snippets, batch_size=batch_size)
        ]

    def score_batch_detailed(
        self,
        snippets: list[str],
        batch_size: int = 64,
    ) -> list[Optional[ScoreResult]]:
        """
        Score a list of code snippets and report truncation.

        Returns one entry per input, in input order: ScoreResult(score,
        truncated), or None for an empty/blank snippet. `truncated` is True when
        the snippet needs more than MAX_TOKENS (512) tokens, <s> and </s>
        included, so only its first 512 were scored.

        Every snippet is tokenized once. Snippets are then ordered by token
        length and cut into batches of at most `batch_size`, each padded only to
        its own longest member (never past 512). The [CLS]-only head and the
        attention mask make this equivalent to padding everything to 512.
        """
        import torch

        if not self._loaded:
            self._load()

        results: list[Optional[ScoreResult]] = [None] * len(snippets)

        valid_indices: list[int] = []
        valid_snippets: list[str] = []
        for i, code in enumerate(snippets):
            if code and str(code).strip():
                valid_indices.append(i)
                valid_snippets.append(str(code))
        if not valid_snippets:
            return results

        token_ids, truncated = self._encode(valid_snippets)
        order = _length_sorted_order([len(ids) for ids in token_ids])
        pad_id = self._tokenizer.pad_token_id

        with torch.inference_mode():
            for start in range(0, len(order), batch_size):
                members = order[start : start + batch_size]
                width = max(len(token_ids[j]) for j in members)

                input_ids      = torch.full((len(members), width), pad_id, dtype=torch.long)
                attention_mask = torch.zeros((len(members), width), dtype=torch.long)
                for row, j in enumerate(members):
                    n = len(token_ids[j])
                    input_ids[row, :n]      = torch.from_numpy(token_ids[j])
                    attention_mask[row, :n] = 1

                logits = self._model(
                    input_ids.to(self._device), attention_mask.to(self._device)
                )
                probs = torch.sigmoid(logits).cpu().numpy().tolist()

                for j, prob in zip(members, probs):
                    results[valid_indices[j]] = ScoreResult(round(float(prob), 4), truncated[j])

        return results

    def _encode(self, texts: list[str]):
        """
        Tokenize each text once. Returns (token id arrays, truncated flags).

        Texts are tokenized with a limit of MAX_TOKENS + 1: one token past the
        window is enough to tell "fits" from "does not fit" without paying to
        tokenize a huge file in full. An over-long sequence is then cut to
        exactly what truncation=True, max_length=512 would give — its first 511
        ids (<s> + 510 code tokens) plus the closing </s>. This relies on the
        RoBERTa layout "<s> … </s>" (pinned by tests against the tokenizer).
        """
        import numpy as np

        token_ids: list = []
        truncated: list[bool] = []
        for start in range(0, len(texts), _TOKENIZE_CHUNK):
            enc = self._tokenizer(
                texts[start : start + _TOKENIZE_CHUNK],
                max_length=MAX_TOKENS + 1,
                truncation=True,
                padding=False,
                return_attention_mask=False,
                verbose=False,
            )
            for ids in enc["input_ids"]:
                if len(ids) > MAX_TOKENS:
                    ids = ids[: MAX_TOKENS - 1] + ids[-1:]
                    truncated.append(True)
                else:
                    truncated.append(False)
                token_ids.append(np.asarray(ids, dtype=np.int32))
        return token_ids, truncated

    # ── Convenience ───────────────────────────────────────────────────────────

    def is_available(self) -> bool:
        """Check whether local model artifacts exist (hub download not yet done)."""
        return (
            (self.model_dir / "best_model.pt").exists() and
            (self.model_dir / "tokenizer").exists()
        )