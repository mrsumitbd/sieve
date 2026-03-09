"""
demo_data/generate.py

Regenerates the bundled demo dataset shipped with SIEVE.
Run once after any changes to the extraction logic that affect the record schema:

    python -m sieve.ui.demo_data.generate

Output files (committed to the repo):
    functions.jsonl   — 54 FunctionRecords across 3 synthetic repos
    classes.jsonl     —  8 ClassRecords across 3 synthetic repos
    manifest.json     — summary metadata
"""

from __future__ import annotations

import json
import sys
from datetime import datetime
from pathlib import Path

# Allow running as a module from the repo root
sys.path.insert(0, str(Path(__file__).resolve().parents[4]))

from sieve.core.extraction import _extract_python

OUT = Path(__file__).parent

# ─── Synthetic source code for three demo "repositories" ─────────────────────

REPOS: dict[str, str] = {}

REPOS["sieve-demo/data-structures"] = """
from __future__ import annotations
from typing import Optional, Generic, TypeVar, Iterator
from dataclasses import dataclass

T = TypeVar('T')


class Node(Generic[T]):
    \"\"\"A node in a singly linked list.\"\"\"

    def __init__(self, value: T, next: Optional[Node[T]] = None):
        self.value = value
        self.next = next


class LinkedList(Generic[T]):
    \"\"\"A singly linked list with O(1) prepend and O(n) append.\"\"\"

    def __init__(self):
        self._head: Optional[Node[T]] = None
        self._size: int = 0

    def prepend(self, value: T) -> None:
        \"\"\"Insert value at the front of the list.\"\"\"
        self._head = Node(value, self._head)
        self._size += 1

    def append(self, value: T) -> None:
        \"\"\"Insert value at the back of the list.\"\"\"
        if self._head is None:
            self._head = Node(value)
            self._size += 1
            return
        cur = self._head
        while cur.next:
            cur = cur.next
        cur.next = Node(value)
        self._size += 1

    def remove(self, value: T) -> bool:
        \"\"\"Remove the first occurrence of value. Returns True if found.\"\"\"
        if self._head is None:
            return False
        if self._head.value == value:
            self._head = self._head.next
            self._size -= 1
            return True
        cur = self._head
        while cur.next:
            if cur.next.value == value:
                cur.next = cur.next.next
                self._size -= 1
                return True
            cur = cur.next
        return False

    def __iter__(self) -> Iterator[T]:
        \"\"\"Iterate over values in insertion order.\"\"\"
        cur = self._head
        while cur:
            yield cur.value
            cur = cur.next

    def __len__(self) -> int:
        return self._size

    def __repr__(self) -> str:
        return "LinkedList([" + ", ".join(str(v) for v in self) + "])"


class Stack(Generic[T]):
    \"\"\"LIFO stack backed by a Python list.\"\"\"

    def __init__(self):
        self._data: list[T] = []

    def push(self, item: T) -> None:
        \"\"\"Push item onto the stack.\"\"\"
        self._data.append(item)

    def pop(self) -> T:
        \"\"\"Remove and return the top item. Raises IndexError if empty.\"\"\"
        if not self._data:
            raise IndexError("pop from empty stack")
        return self._data.pop()

    def peek(self) -> T:
        \"\"\"Return top item without removing it.\"\"\"
        if not self._data:
            raise IndexError("peek at empty stack")
        return self._data[-1]

    def is_empty(self) -> bool:
        return len(self._data) == 0

    def __len__(self) -> int:
        return len(self._data)


class Queue(Generic[T]):
    \"\"\"FIFO queue backed by a deque for O(1) enqueue and dequeue.\"\"\"

    def __init__(self):
        from collections import deque
        self._data: deque[T] = deque()

    def enqueue(self, item: T) -> None:
        \"\"\"Add item to the back of the queue.\"\"\"
        self._data.append(item)

    def dequeue(self) -> T:
        \"\"\"Remove and return the front item. Raises IndexError if empty.\"\"\"
        if not self._data:
            raise IndexError("dequeue from empty queue")
        return self._data.popleft()

    def front(self) -> T:
        \"\"\"Return front item without removing it.\"\"\"
        if not self._data:
            raise IndexError("front of empty queue")
        return self._data[0]

    def is_empty(self) -> bool:
        return len(self._data) == 0

    def __len__(self) -> int:
        return len(self._data)


@dataclass
class TreeNode(Generic[T]):
    \"\"\"A node in a binary search tree.\"\"\"
    value: T
    left: Optional[TreeNode[T]] = None
    right: Optional[TreeNode[T]] = None


class BinarySearchTree(Generic[T]):
    \"\"\"Binary search tree with in-order traversal and iterative search.\"\"\"

    def __init__(self):
        self._root: Optional[TreeNode[T]] = None

    def insert(self, value: T) -> None:
        \"\"\"Insert a value into the BST.\"\"\"
        if self._root is None:
            self._root = TreeNode(value)
            return
        cur = self._root
        while True:
            if value < cur.value:
                if cur.left is None:
                    cur.left = TreeNode(value)
                    return
                cur = cur.left
            else:
                if cur.right is None:
                    cur.right = TreeNode(value)
                    return
                cur = cur.right

    def contains(self, value: T) -> bool:
        \"\"\"Return True if value exists in the BST.\"\"\"
        cur = self._root
        while cur:
            if value == cur.value:
                return True
            cur = cur.left if value < cur.value else cur.right
        return False

    def inorder(self) -> list[T]:
        \"\"\"Return sorted list via in-order traversal.\"\"\"
        result: list[T] = []

        def _walk(node: Optional[TreeNode[T]]) -> None:
            if node:
                _walk(node.left)
                result.append(node.value)
                _walk(node.right)

        _walk(self._root)
        return result
"""

REPOS["sieve-demo/algorithms"] = """
from typing import TypeVar
import heapq

T = TypeVar("T")


def binary_search(arr: list, target, lo: int = 0, hi: int = None) -> int:
    \"\"\"
    Iterative binary search.
    Returns index of target in arr, or -1 if not found.
    Assumes arr is sorted in ascending order.
    \"\"\"
    if hi is None:
        hi = len(arr) - 1
    while lo <= hi:
        mid = (lo + hi) // 2
        if arr[mid] == target:
            return mid
        elif arr[mid] < target:
            lo = mid + 1
        else:
            hi = mid - 1
    return -1


def merge_sort(arr: list) -> list:
    \"\"\"
    Top-down merge sort. Returns a new sorted list.
    Time: O(n log n), Space: O(n).
    \"\"\"
    if len(arr) <= 1:
        return arr[:]
    mid = len(arr) // 2
    left = merge_sort(arr[:mid])
    right = merge_sort(arr[mid:])
    return _merge(left, right)


def _merge(left: list, right: list) -> list:
    result = []
    i = j = 0
    while i < len(left) and j < len(right):
        if left[i] <= right[j]:
            result.append(left[i])
            i += 1
        else:
            result.append(right[j])
            j += 1
    result.extend(left[i:])
    result.extend(right[j:])
    return result


def quicksort(arr: list, lo: int = 0, hi: int = None) -> None:
    \"\"\"
    In-place quicksort using Lomuto partition scheme.
    Mutates arr in place.
    \"\"\"
    if hi is None:
        hi = len(arr) - 1
    if lo < hi:
        p = _partition(arr, lo, hi)
        quicksort(arr, lo, p - 1)
        quicksort(arr, p + 1, hi)


def _partition(arr: list, lo: int, hi: int) -> int:
    pivot = arr[hi]
    i = lo - 1
    for j in range(lo, hi):
        if arr[j] <= pivot:
            i += 1
            arr[i], arr[j] = arr[j], arr[i]
    arr[i + 1], arr[hi] = arr[hi], arr[i + 1]
    return i + 1


def heapsort(arr: list) -> list:
    \"\"\"
    Heapsort using Python's heapq module.
    Returns a new sorted list without mutating the input.
    \"\"\"
    h = arr[:]
    heapq.heapify(h)
    return [heapq.heappop(h) for _ in range(len(h))]


def is_sorted(arr: list, reverse: bool = False) -> bool:
    \"\"\"Return True if arr is sorted (ascending by default).\"\"\"
    if len(arr) < 2:
        return True
    pairs = zip(arr, arr[1:])
    return all(a >= b for a, b in pairs) if reverse else all(a <= b for a, b in pairs)


def count_inversions(arr: list) -> int:
    \"\"\"
    Count inversions in arr using merge sort.
    An inversion is a pair (i, j) where i < j but arr[i] > arr[j].
    \"\"\"
    _, count = _count_inversions_helper(arr)
    return count


def _count_inversions_helper(arr: list) -> tuple[list, int]:
    if len(arr) <= 1:
        return arr[:], 0
    mid = len(arr) // 2
    left, lc = _count_inversions_helper(arr[:mid])
    right, rc = _count_inversions_helper(arr[mid:])
    merged, mc = _merge_count(left, right)
    return merged, lc + rc + mc


def _merge_count(left: list, right: list) -> tuple[list, int]:
    result = []
    count = 0
    i = j = 0
    while i < len(left) and j < len(right):
        if left[i] <= right[j]:
            result.append(left[i])
            i += 1
        else:
            result.append(right[j])
            count += len(left) - i
            j += 1
    result.extend(left[i:])
    result.extend(right[j:])
    return result, count
"""

REPOS["sieve-demo/text-utils"] = """
import re
from typing import Optional
from collections import Counter


def normalize_whitespace(text: str) -> str:
    \"\"\"Replace all whitespace sequences with a single space and strip.\"\"\"
    return re.sub(r"\\s+", " ", text).strip()


def remove_punctuation(text: str, keep: str = "") -> str:
    \"\"\"Remove punctuation from text. Optionally keep specific characters.\"\"\"
    pattern = f"[^\\\\w\\\\s{re.escape(keep)}]"
    return re.sub(pattern, "", text)


def to_snake_case(name: str) -> str:
    \"\"\"Convert CamelCase or PascalCase identifier to snake_case.\"\"\"
    s1 = re.sub(r"(.)([A-Z][a-z]+)", r"\\1_\\2", name)
    return re.sub(r"([a-z0-9])([A-Z])", r"\\1_\\2", s1).lower()


def to_camel_case(name: str) -> str:
    \"\"\"Convert snake_case identifier to camelCase.\"\"\"
    parts = name.split("_")
    return parts[0] + "".join(p.capitalize() for p in parts[1:])


def truncate(text: str, max_len: int, suffix: str = "...") -> str:
    \"\"\"Truncate text to max_len characters, appending suffix if truncated.\"\"\"
    if len(text) <= max_len:
        return text
    return text[:max_len - len(suffix)] + suffix


def word_count(text: str) -> int:
    \"\"\"Return the number of words in text.\"\"\"
    return len(text.split())


def sentence_count(text: str) -> int:
    \"\"\"Count sentences in text using terminal punctuation as delimiter.\"\"\"
    return len(re.findall(r"[.!?]+", text))


def most_common_words(
    text: str, n: int = 10, stopwords: set = None
) -> list[tuple[str, int]]:
    \"\"\"
    Return the n most common words in text as (word, count) pairs.
    Optionally filter out stopwords.
    \"\"\"
    words = re.findall(r"\\b\\w+\\b", text.lower())
    if stopwords:
        words = [w for w in words if w not in stopwords]
    return Counter(words).most_common(n)


def levenshtein(s: str, t: str) -> int:
    \"\"\"
    Compute Levenshtein edit distance using Wagner-Fischer DP.
    \"\"\"
    m, n = len(s), len(t)
    dp = list(range(n + 1))
    for i in range(1, m + 1):
        prev = dp[:]
        dp[0] = i
        for j in range(1, n + 1):
            if s[i - 1] == t[j - 1]:
                dp[j] = prev[j - 1]
            else:
                dp[j] = 1 + min(prev[j], dp[j - 1], prev[j - 1])
    return dp[n]


def is_palindrome(text: str, ignore_case: bool = True, ignore_spaces: bool = True) -> bool:
    \"\"\"Return True if text is a palindrome.\"\"\"
    s = text
    if ignore_case:
        s = s.lower()
    if ignore_spaces:
        s = s.replace(" ", "")
    return s == s[::-1]


class Tokenizer:
    \"\"\"
    Simple whitespace tokenizer with optional lowercasing and stopword removal.
    \"\"\"

    def __init__(self, lowercase: bool = True, stopwords: Optional[set] = None):
        self.lowercase = lowercase
        self.stopwords = stopwords or set()

    def tokenize(self, text: str) -> list[str]:
        \"\"\"Split text into tokens, applying configured transforms.\"\"\"
        if self.lowercase:
            text = text.lower()
        tokens = re.findall(r"\\b\\w+\\b", text)
        if self.stopwords:
            tokens = [t for t in tokens if t not in self.stopwords]
        return tokens

    def tokenize_batch(self, texts: list[str]) -> list[list[str]]:
        \"\"\"Tokenize a list of texts.\"\"\"
        return [self.tokenize(t) for t in texts]

    def vocabulary(self, texts: list[str]) -> set[str]:
        \"\"\"Return the full vocabulary across a list of texts.\"\"\"
        vocab: set[str] = set()
        for text in texts:
            vocab.update(self.tokenize(text))
        return vocab


class TextCleaner:
    \"\"\"
    Configurable text cleaning pipeline with chainable transforms.
    \"\"\"

    def __init__(self):
        self._steps: list = []

    def add_step(self, fn) -> TextCleaner:
        \"\"\"Add a transform function to the pipeline. Returns self for chaining.\"\"\"
        self._steps.append(fn)
        return self

    def clean(self, text: str) -> str:
        \"\"\"Apply all pipeline steps in order.\"\"\"
        for step in self._steps:
            text = step(text)
        return text

    def clean_batch(self, texts: list[str]) -> list[str]:
        \"\"\"Apply pipeline to a list of texts.\"\"\"
        return [self.clean(t) for t in texts]

    @classmethod
    def default(cls) -> TextCleaner:
        \"\"\"Return a cleaner with sensible defaults: whitespace normalization.\"\"\"
        return cls().add_step(normalize_whitespace)
"""

REPO_META = {
    "sieve-demo/data-structures": {"stars": 1240, "contributors": 12, "license": "MIT"},
    "sieve-demo/algorithms":      {"stars":  890, "contributors":  8, "license": "MIT"},
    "sieve-demo/text-utils":      {"stars":  430, "contributors":  5, "license": "Apache-2.0"},
}


def _serialize(record) -> dict:
    """Convert a FunctionRecord or ClassRecord to a JSON-serializable dict."""
    import dataclasses
    d = dataclasses.asdict(record)
    # Ensure list fields that may be None are empty lists
    for key in ("parameters", "used_imports", "decorators",
                "parent_classes", "method_names"):
        if key in d and d[key] is None:
            d[key] = []
    d["llm_score"] = None
    return d


def generate() -> None:
    all_funcs, all_classes = [], []
    repo_stats = []

    for repo_name, code in REPOS.items():
        funcs, classes = _extract_python(code, "main.py", repo_name)
        all_funcs.extend(funcs)
        all_classes.extend(classes)
        meta = REPO_META[repo_name]
        repo_stats.append({
            "repo": repo_name,
            "stars": meta["stars"],
            "contributors": meta["contributors"],
            "functions": len(funcs),
            "classes": len(classes),
            "test_suite_present": True,
            "test_confidence": "high",
            "license": meta["license"],
        })

    # Write JSONL files
    fn_path = OUT / "functions.jsonl"
    cl_path = OUT / "classes.jsonl"
    mn_path = OUT / "manifest.json"

    fn_path.write_text(
        "\n".join(json.dumps(_serialize(f)) for f in all_funcs) + "\n",
        encoding="utf-8",
    )
    cl_path.write_text(
        "\n".join(json.dumps(_serialize(c)) for c in all_classes) + "\n",
        encoding="utf-8",
    )

    manifest = {
        "generated_at": datetime.utcnow().isoformat() + "Z",
        "description": "Bundled demo dataset for SIEVE — synthetic repos, real extraction.",
        "repos": list(REPOS.keys()),
        "summary": {
            "total_functions": len(all_funcs),
            "total_classes": len(all_classes),
        },
        "repo_stats": repo_stats,
    }
    mn_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    print(f"Generated {len(all_funcs)} functions, {len(all_classes)} classes")
    for r in repo_stats:
        print(f"  {r['repo']}: {r['functions']} funcs, {r['classes']} classes")
    print(f"Written to {OUT}")


if __name__ == "__main__":
    generate()