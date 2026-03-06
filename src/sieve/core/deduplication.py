"""
core/deduplication.py

Near-duplicate detection using MinHash LSH (datasketch).
Operates on source code strings — tokenizes at the word/token level
rather than character level for better code similarity matching.
"""

import re
import logging
from datasketch import MinHash, MinHashLSH

logger = logging.getLogger(__name__)


def _tokenize_code(source: str) -> list[str]:
    """
    Tokenize source code into a bag of tokens for MinHash.
    Strips string literals and normalizes identifiers to reduce
    surface-level variations (variable renames, whitespace).
    """
    # Normalize whitespace
    source = re.sub(r"\s+", " ", source)
    # Remove string literals — content varies but structure is what matters
    source = re.sub(r'""".*?"""', '"""STR"""', source, flags=re.DOTALL)
    source = re.sub(r"'''.*?'''", "'''STR'''", source, flags=re.DOTALL)
    source = re.sub(r'"[^"]*"', '"STR"', source)
    source = re.sub(r"'[^']*'", "'STR'", source)
    # Tokenize on word boundaries and punctuation
    tokens = re.findall(r"[a-zA-Z_][a-zA-Z0-9_]*|[^\s\w]", source)
    return tokens


def _make_minhash(source: str, num_perm: int = 128) -> MinHash:
    m = MinHash(num_perm=num_perm)
    for token in _tokenize_code(source):
        m.update(token.encode("utf-8"))
    return m


def deduplicate(
    items: list,
    source_code_attr: str = "source_code",
    threshold: float = 0.8,
    num_perm: int = 128,
) -> list:
    """
    Remove near-duplicate records from a list of FunctionRecord or ClassRecord objects.

    Args:
        items: List of records with a source_code attribute
        source_code_attr: Attribute name containing the source code string
        threshold: Jaccard similarity threshold above which two items are considered duplicates
        num_perm: Number of permutations for MinHash (higher = more accurate, slower)

    Returns:
        Deduplicated list, keeping the first occurrence of each near-duplicate cluster
    """
    if not items:
        return items

    lsh = MinHashLSH(threshold=threshold, num_perm=num_perm)
    kept = []
    skipped = 0

    for idx, item in enumerate(items):
        source = getattr(item, source_code_attr, "")
        if not source:
            kept.append(item)
            continue

        m = _make_minhash(source, num_perm)
        key = f"item_{idx}"

        try:
            result = lsh.query(m)
            if result:
                # Near-duplicate found — skip this item
                skipped += 1
                continue
            lsh.insert(key, m)
            kept.append(item)
        except Exception as e:
            logger.warning(f"Dedup error at index {idx}: {e}")
            kept.append(item)

    logger.info(f"Deduplication: {len(items)} → {len(kept)} items ({skipped} duplicates removed)")
    return kept
