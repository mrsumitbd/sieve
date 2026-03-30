"""
sieve/models/classifier.py

Wraps the fine-tuned CodeBERT classifier for use inside the SIEVE pipeline.
Scores each extracted code snippet with P(AI-generated) → llm_score field.

Usage:
    from sieve.models.classifier import LLMCodeClassifier

    clf = LLMCodeClassifier.load()  # loads from default path
    scores = clf.score_batch(["def foo(): ...", "public void bar() {}"])
"""

import logging
from pathlib import Path
from typing import Optional, Union

logger = logging.getLogger(__name__)

# Default model artifacts location — relative to this file's parent (src/sieve/models/)
_DEFAULT_MODEL_DIR = Path(__file__).parent / "artifacts"


class LLMCodeClassifier:
    """
    Binary classifier: P(code is AI-generated).

    Wraps microsoft/codebert-base fine-tuned on 114K human/AI code pairs
    across Python, Java, JavaScript, and C++.
    """

    def __init__(self, model_dir: Union[str, Path]):
        self.model_dir  = Path(model_dir)
        self._model     = None
        self._tokenizer = None
        self._device    = None
        self._loaded    = False

    # ── Loading ───────────────────────────────────────────────────────────────

    @classmethod
    def load(cls, model_dir: Optional[Union[str, Path]] = None) -> "LLMCodeClassifier":
        """
        Load classifier from model_dir (defaults to src/sieve/models/artifacts/).
        Raises FileNotFoundError if artifacts are missing.
        """
        if model_dir is None:
            model_dir = _DEFAULT_MODEL_DIR

        model_dir = Path(model_dir)
        weights_path = model_dir / "best_model.pt"
        tokenizer_path = model_dir / "tokenizer"

        if not weights_path.exists():
            raise FileNotFoundError(
                f"Model weights not found at {weights_path}.\n"
                f"Run scripts/classifier/train.py first, then copy artifacts:\n"
                f"  cp -r data/models/* src/sieve/models/artifacts/"
            )
        if not tokenizer_path.exists():
            raise FileNotFoundError(
                f"Tokenizer not found at {tokenizer_path}.\n"
                f"Run scripts/classifier/train.py first, then copy artifacts:\n"
                f"  cp -r data/models/* src/sieve/models/artifacts/"
            )

        instance = cls(model_dir)
        instance._load()
        return instance

    def _load(self):
        """Lazy-load model and tokenizer onto device."""
        import torch
        import torch.nn as nn
        from transformers import AutoModel, AutoTokenizer

        # Device selection
        if torch.cuda.is_available():
            self._device = torch.device("cuda")
        elif torch.backends.mps.is_available():
            self._device = torch.device("mps")
        else:
            self._device = torch.device("cpu")
        logger.info(f"LLMCodeClassifier: using device {self._device}")

        # Tokenizer
        self._tokenizer = AutoTokenizer.from_pretrained(
            str(self.model_dir / "tokenizer"),
            local_files_only=True,
        )

        # Model — recreate same architecture as training
        class _CodeBERTClassifier(nn.Module):
            def __init__(self):
                super().__init__()
                self.encoder    = AutoModel.from_pretrained(
                    "microsoft/codebert-base",
                    local_files_only=False,
                )
                self.dropout    = nn.Dropout(0.1)
                self.classifier = nn.Linear(self.encoder.config.hidden_size, 1)

            def forward(self, input_ids, attention_mask):
                out = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
                cls = out.last_hidden_state[:, 0, :]
                cls = self.dropout(cls)
                return self.classifier(cls).squeeze(-1)

        self._model = _CodeBERTClassifier().to(self._device)
        state = torch.load(
            self.model_dir / "best_model.pt",
            map_location=self._device,
        )
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
        Returns list of P(AI-generated) ∈ [0, 1] (None for empty snippets).
        """
        import torch

        if not self._loaded:
            self._load()

        scores: list[Optional[float]] = []
        valid_indices = []
        valid_snippets = []

        for i, code in enumerate(snippets):
            if code and str(code).strip():
                valid_indices.append(i)
                valid_snippets.append(str(code))
            else:
                scores.append(None)

        # Fill scores list with placeholders for valid entries
        scores = [None] * len(snippets)

        # Process in batches
        all_probs = []
        for start in range(0, len(valid_snippets), batch_size):
            batch = valid_snippets[start : start + batch_size]
            enc = self._tokenizer(
                batch,
                max_length=512,
                padding="max_length",
                truncation=True,
                return_tensors="pt",
            )
            input_ids      = enc["input_ids"].to(self._device)
            attention_mask = enc["attention_mask"].to(self._device)

            with torch.no_grad():
                logits = self._model(input_ids, attention_mask)
                probs  = torch.sigmoid(logits).cpu().numpy().tolist()
                all_probs.extend(probs)

        for idx, prob in zip(valid_indices, all_probs):
            scores[idx] = round(float(prob), 4)

        return scores

    # ── Convenience ───────────────────────────────────────────────────────────

    def is_available(self) -> bool:
        """Check whether model artifacts exist without loading them."""
        return (
            (self.model_dir / "best_model.pt").exists() and
            (self.model_dir / "tokenizer").exists()
        )