"""
tests/tiny_roberta.py

Hermetic stand-in for CodeBERT: a tiny, randomly initialised RoBERTa encoder plus
a word-level tokenizer built locally. No network, no real weights.

Imports of torch / transformers / tokenizers are lazy so that importing this
module never fails when they are absent; callers skip via pytest.importorskip.

Tokenizer contract (exactly one token per whitespace-separated word):
    "w1 w2 w3"  ->  [<s>, w1, w2, w3, </s>]   (n words -> n + 2 ids)
so a test can build a snippet with a precise total token count.
"""

from __future__ import annotations

import random
from pathlib import Path

MAX_TOKENS = 512
N_WORDS = 64
WORDS = [f"w{i}" for i in range(N_WORDS)]

_SPECIALS = ["<s>", "<pad>", "</s>", "<unk>"]   # ids 0, 1, 2, 3 — same as RoBERTa
PAD_ID = 1


def snippet_with_total_tokens(total: int, seed: int = 0) -> str:
    """A snippet that tokenizes to exactly `total` ids, special tokens included."""
    assert total >= 3
    rng = random.Random(seed)
    return " ".join(rng.choice(WORDS) for _ in range(total - 2))


def make_tokenizer():
    from tokenizers import Tokenizer, models, pre_tokenizers, processors
    from transformers import PreTrainedTokenizerFast

    vocab = {tok: i for i, tok in enumerate(_SPECIALS + WORDS)}
    tok = Tokenizer(models.WordLevel(vocab, unk_token="<unk>"))
    tok.pre_tokenizer = pre_tokenizers.WhitespaceSplit()
    # Same ids as RobertaProcessing ("<s> A </s>", pairs "<s> A </s> </s> B </s>"), but
    # via TemplateProcessing: RobertaProcessing renamed its `cls` keyword to `cls_token`
    # in tokenizers 0.23, which broke keyword calls. This interface is stable.
    tok.post_processor = processors.TemplateProcessing(
        single="<s> $A </s>",
        pair="<s> $A </s> </s> $B </s>",
        special_tokens=[("<s>", 0), ("</s>", 2)],
    )
    return PreTrainedTokenizerFast(
        tokenizer_object=tok,
        bos_token="<s>", eos_token="</s>", sep_token="</s>", cls_token="<s>",
        pad_token="<pad>", unk_token="<unk>",
    )


def make_encoder(seed: int = 0):
    """Tiny RobertaModel, re-initialised with a large std so scores really vary."""
    import torch
    from transformers import RobertaConfig, RobertaModel

    torch.manual_seed(seed)
    cfg = RobertaConfig(
        vocab_size=len(_SPECIALS) + N_WORDS,
        hidden_size=32, num_hidden_layers=2, num_attention_heads=2,
        intermediate_size=64,
        max_position_embeddings=MAX_TOKENS + 2,    # RoBERTa offsets positions by pad_id + 1
        type_vocab_size=1,
        pad_token_id=PAD_ID, bos_token_id=0, eos_token_id=2,
        hidden_dropout_prob=0.1, attention_probs_dropout_prob=0.1,
    )
    encoder = RobertaModel(cfg)                   # keeps the pooler, as AutoModel does
    with torch.no_grad():
        for name, p in encoder.named_parameters():
            if p.dim() > 1:
                p.normal_(0.0, 0.6)
    return encoder


def make_head_model(encoder):
    """
    Test-local copy of the production head (encoder -> [CLS] -> dropout -> linear).
    The production class is checked against this one in test_classifier_batching.py.
    """
    import torch.nn as nn

    class _Head(nn.Module):
        def __init__(self):
            super().__init__()
            self.encoder = encoder
            self.dropout = nn.Dropout(0.1)
            self.classifier = nn.Linear(encoder.config.hidden_size, 1)

        def forward(self, input_ids, attention_mask):
            out = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
            cls = out.last_hidden_state[:, 0, :]
            return self.classifier(self.dropout(cls)).squeeze(-1)

    return _Head()


def make_classifier(model_dir: Path, seed: int = 0, device: str = "cpu"):
    """A fully loaded LLMCodeClassifier backed by the tiny model — no _load() call."""
    import torch
    from sieve.models.classifier import LLMCodeClassifier

    torch.manual_seed(seed)
    model = make_head_model(make_encoder(seed))
    with torch.no_grad():
        model.classifier.weight.normal_(0.0, 1.0)
        model.classifier.bias.zero_()
    model.eval()

    clf = LLMCodeClassifier(model_dir)
    clf._tokenizer = make_tokenizer()
    clf._model = model.to(device)
    clf._device = torch.device(device)
    clf._loaded = True
    return clf


def old_scores(clf, snippets, batch_size: int = 64):
    """
    The pre-change algorithm, verbatim: pad every snippet to 512, in input
    order, under no_grad. Used as the reference for the equivalence test.
    """
    import torch

    valid = [(i, s) for i, s in enumerate(snippets) if s and str(s).strip()]
    out = [None] * len(snippets)
    probs = []
    for start in range(0, len(valid), batch_size):
        batch = [s for _, s in valid[start:start + batch_size]]
        enc = clf._tokenizer(
            batch, max_length=MAX_TOKENS, padding="max_length",
            truncation=True, return_tensors="pt",
        )
        with torch.no_grad():
            logits = clf._model(enc["input_ids"].to(clf._device),
                                enc["attention_mask"].to(clf._device))
            probs.extend(torch.sigmoid(logits).cpu().numpy().tolist())
    for (i, _), p in zip(valid, probs):
        out[i] = round(float(p), 4)
    return out
