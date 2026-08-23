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
"""

import logging
from pathlib import Path
from typing import Optional, Union

logger = logging.getLogger(__name__)

# HuggingFace Hub repo containing model weights
HF_REPO_ID = "mrahman2025/sieve-llm-classifier"

# Default local cache location
_DEFAULT_MODEL_DIR = Path(__file__).parent / "artifacts"


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
        self._loaded    = False

    # ── Loading ───────────────────────────────────────────────────────────────

    @classmethod
    def load(
        cls,
        model_dir: Optional[Union[str, Path]] = None,
        hf_token: Optional[str] = None,
    ) -> "LLMCodeClassifier":
        """
        Load classifier. Uses HuggingFace Hub cache by default.
        Falls back to local artifacts if model_dir is provided and exists.

        Args:
            model_dir: Optional local path to model artifacts directory.
            hf_token:  Optional HuggingFace token for private repos.
        """
        instance = cls(model_dir or _DEFAULT_MODEL_DIR)
        instance._hf_token = hf_token
        instance._load()
        return instance

    def _load(self):
        """Load model and tokenizer — from HF Hub cache or local artifacts."""
        import torch
        import torch.nn as nn
        from transformers import AutoModel, AutoTokenizer
        from huggingface_hub import hf_hub_download

        # Device selection
        if torch.cuda.is_available():
            self._device = torch.device("cuda")
        elif torch.backends.mps.is_available():
            self._device = torch.device("mps")
        else:
            self._device = torch.device("cpu")
        logger.info(f"LLMCodeClassifier: using device {self._device}")

        hf_token = getattr(self, "_hf_token", None)

        # ── Tokenizer: load from HF Hub (cached to ~/.cache/huggingface) ──────
        local_tok = self.model_dir / "tokenizer"
        if local_tok.exists():
            logger.info("Loading tokenizer from local artifacts...")
            tok_path = str(local_tok)
        else:
            logger.info(f"Loading tokenizer from HF Hub ({HF_REPO_ID})...")
            tok_path = HF_REPO_ID

        self._tokenizer = AutoTokenizer.from_pretrained(
            tok_path,
            token=hf_token,
        )

        # ── Model weights: load from HF Hub cache ─────────────────────────────
        local_weights = self.model_dir / "best_model.pt"
        if local_weights.exists():
            logger.info("Loading weights from local artifacts...")
            weights_path = local_weights
        else:
            logger.info(f"Downloading weights from HF Hub ({HF_REPO_ID})...")
            weights_path = hf_hub_download(
                repo_id=HF_REPO_ID,
                filename="best_model.pt",
                token=hf_token,
            )

        # ── Model architecture ────────────────────────────────────────────────
        class _CodeBERTClassifier(nn.Module):
            def __init__(self):
                super().__init__()
                self.encoder    = AutoModel.from_pretrained(
                    "microsoft/codebert-base",
                )
                self.dropout    = nn.Dropout(0.1)
                self.classifier = nn.Linear(self.encoder.config.hidden_size, 1)

            def forward(self, input_ids, attention_mask):
                out = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
                cls = out.last_hidden_state[:, 0, :]
                cls = self.dropout(cls)
                return self.classifier(cls).squeeze(-1)

        self._model = _CodeBERTClassifier().to(self._device)
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
        """Check whether local model artifacts exist (hub download not yet done)."""
        return (
            (self.model_dir / "best_model.pt").exists() and
            (self.model_dir / "tokenizer").exists()
        )