# sieve/models/

This directory holds the serialized LLM-generated code classifier.

## Expected artifact

`llm_classifier.joblib` — a trained binary classifier that outputs
P(LLM-generated) for a given source code snippet.

## Status

**Not yet built.** The `llm_score` field on every `FunctionRecord` and
`ClassRecord` will remain `null` until this classifier is trained and
placed here.

## Planned approach

- **Training data**: CodeProbe dataset (human-written vs. LLM-generated
  code, labeled, multi-language)
- **Features**: 72 interpretable static features from CodeProbe
  (complexity, naming conventions, comment density, structural patterns)
- **Architecture**: Gradient-boosted classifier (XGBoost or LightGBM) —
  fast at inference, no GPU required, interpretable via SHAP
- **Interface**: `sieve/models/classifier.py` will expose a
  `LLMCodeClassifier` class with a `.score(source_code: str) -> float`
  method

## Integration point

Once built, uncomment the following block in `sieve/pipeline.py`
(Phase 5 — LLM score annotation):

```python
from sieve.models.classifier import LLMCodeClassifier
clf = LLMCodeClassifier.load("sieve/models/llm_classifier.joblib")
for record in all_functions + all_classes:
    record.llm_score = clf.score(record.source_code)
```
