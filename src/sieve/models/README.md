# sieve/models/

`classifier.py` defines `LLMCodeClassifier`, a CodeBERT-based binary classifier
that returns P(LLM-generated) for a code snippet. It fills the `llm_score`
field on function and class records.

## Weights

- Weights (`best_model.pt`) are hosted on the Hugging Face Hub at
  `HF_REPO_ID = "mrahman2025/sieve-llm-classifier"`.
- They are downloaded lazily with `hf_hub_download` on first use and cached by
  `huggingface_hub`.
- If `artifacts/best_model.pt` exists locally, it is used instead of downloading.
  The `artifacts/` weights and tokenizer files are git-ignored.
- The tokenizer and encoder come from `microsoft/codebert-base`.

## Interface

- `LLMCodeClassifier.load(model_dir=None, hf_token=None)` loads the model.
- `score_batch(snippets, batch_size=64)` returns one score in [0, 1] per snippet,
  or `None` for an empty snippet. `score(code)` scores a single snippet.

## Pipeline use

Scoring is Phase 5 of `sieve/pipeline.py` and runs only when
`annotate_llm_score` is enabled (default `False`). A failure to load or score is
non-fatal: the error is reported through the progress callback and the run
continues. `llm_score` is `null` when annotation is off or failed.

`hf_token` is optional and is set only through the config or API (there is no
CLI option for it).

## Dependencies

`torch`, `transformers`, `huggingface_hub` and `sentencepiece` are listed in
`requirements.txt`, not in the `dependencies` of `pyproject.toml`.
