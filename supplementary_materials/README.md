# Supplementary materials

This folder holds material that accompanies SIEVE. It is not part of the `sieve` package and SIEVE does not read it.

For the evaluation these records were used in, see the SIEVE preprint: [arXiv:2610.22544](https://arxiv.org/abs/2610.22544).

## Contents

| Path | What it is |
|------|------------|
| [`validation_samples/`](validation_samples/) | 400 records extracted by SIEVE and used in a manual validation of its extraction. |

## Validation samples

`validation_samples/` has 400 records, 100 for each of four languages (Python, Java, JavaScript, C++). For each language, 50 are function/method records and 50 are class records. The manual ratings are not included.

The records were produced by an early build of SIEVE, so field names and layout can differ from the current release. For example, current records also carry an `llm_score_truncated` field, which these do not. For the current layout see *Output Schema* in the [root README](../README.md#output-schema) and the *Record Schema* section of the Documentation page in the web UI.

### Layout

```
validation_samples/
├── SHA256SUMS
├── CPP/
│   ├── cpp_classes.jsonl
│   └── cpp_functions.jsonl
├── JAVA/
│   ├── java_classes.jsonl
│   └── java_functions.jsonl
├── JS/
│   ├── js_classes.jsonl
│   └── js_functions.jsonl
└── Python/
    ├── python_classes.jsonl
    └── python_functions.jsonl
```

The files are plain JSONL: one JSON object per line, 50 records per file. They total about 17 MB because each record includes its full AST.

### Record fields

Function/method files (`*_functions.jsonl`) and class files (`*_classes.jsonl`) share most fields. The groups below are the field names as they appear in these files.

**Identification** (both record types)

| Field | Meaning |
|-------|---------|
| `repo` | Source repository, as `owner/name`. |
| `file_path` | Path of the file inside the repository. |
| `language` | `Python`, `Java`, `JavaScript` or `C++`. |
| `commit_sha` | The exact commit the record was extracted from. |
| `start_line`, `end_line` | Line range of the function or class in that file at that commit (1-based, inclusive). |

**Shared content fields** (both record types)

`docstring`, `source_code`, `used_imports`, `decorators`.

**Function-only fields** (`*_functions.jsonl`)

`func_name`, `parameters`, `return_annotation`, `signature`, `is_method`, `parent_class`.

**Class-only fields** (`*_classes.jsonl`)

`class_name`, `parent_classes`, `skeleton`, `method_names`, `method_count`, `has_constructor`.

**Metrics** (both record types)

- Size and comments: `loc`, `sloc`, `lloc`, `comments`, `multi`, `blank`, `comment_ratio`.
- Complexity: `cyclomatic_complexity`, `max_nesting_depth`.
- Halstead measures: `h1`, `h2`, `N1`, `N2`, `vocabulary`, `halstead_length`, `calculated_length`, `volume`, `difficulty`, `effort`, `time`, `bugs`.
- `maintainability_index`.

See *Code Metrics* in the [root README](../README.md#code-metrics) for what these are.

**AST** (both record types)

| Field | Meaning |
|-------|---------|
| `ast_depth` | Depth of the syntax tree. |
| `ast_num_nodes` | Number of nodes in the syntax tree. |
| `ast_node_types` | Count of nodes per node type. |
| `ast` | The full syntax tree, as nested objects with `type`, `text` and `children`. This is what makes the files large. |

**`llm_score`** (both record types)

SIEVE's AI-generated-code score for the record. See *AI-Generated-Code Score* in the [root README](../README.md#ai-generated-code-score).

### Reading the records

Run from this folder (`supplementary_materials/`):

```python
import json

def read_jsonl(path):
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]

records = read_jsonl("validation_samples/Python/python_functions.jsonl")
print(len(records))                      # 50
r = records[0]
print(r["repo"], r["func_name"], r["start_line"], r["end_line"])
print(r["source_code"])
```

### Verifying integrity

`SHA256SUMS` lists a SHA-256 checksum for each `.jsonl` file. From this folder:

```bash
cd validation_samples && shasum -a 256 -c SHA256SUMS
```

Every line should end in `OK`.

### Opening a record on GitHub

Each record names its repository and the exact commit it came from, so it can be opened at the right place in the source:

```
https://github.com/<repo>/blob/<commit_sha>/<file_path>#L<start_line>-L<end_line>
```

In Python:

```python
url = (
    f"https://github.com/{r['repo']}/blob/{r['commit_sha']}"
    f"/{r['file_path']}#L{r['start_line']}-L{r['end_line']}"
)
```

If a repository has since been renamed, made private or deleted, or its history rewritten, the link may no longer resolve. The record's `source_code` field still holds the excerpt.

### Licensing

Each record is an excerpt of a third-party open-source repository and remains under that repository's license. The record names the repository (`repo`) and the exact commit (`commit_sha`) so the license that applies can be looked up. The MIT license in this repository's [`LICENSE`](../LICENSE) covers SIEVE's own code, not these excerpts.
