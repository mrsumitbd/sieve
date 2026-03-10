"""
conftest.py

Shared pytest fixtures for the SIEVE test suite.
"""

import json
import textwrap
import tempfile
from pathlib import Path

import pytest


# ─── Source code fixtures ─────────────────────────────────────────────────────

@pytest.fixture
def python_source() -> str:
    return textwrap.dedent("""
        import os
        import re
        from pathlib import Path
        from typing import Optional, List
        from dataclasses import dataclass
        from functools import lru_cache

        @dataclass
        class Config:
            \"\"\"Application configuration.\"\"\"
            name: str
            debug: bool = False

        class FileProcessor:
            \"\"\"Processes files in a directory.\"\"\"

            def __init__(self, base_dir: str):
                self.base_dir = Path(base_dir)

            def process(self, pattern: str = '*.txt') -> int:
                \"\"\"Process all files matching pattern.\"\"\"
                count = 0
                for f in self.base_dir.glob(pattern):
                    self._handle(f)
                    count += 1
                return count

            def _handle(self, path: Path) -> Optional[str]:
                content = path.read_text()
                result = re.sub(r'\\s+', ' ', content)
                return result

        @staticmethod
        @lru_cache(maxsize=128)
        def find_files(directory: str, ext: str) -> List[str]:
            \"\"\"Find all files with given extension.\"\"\"
            return [str(p) for p in Path(directory).rglob(f'*.{ext}')]

        def normalize(text: str) -> str:
            return re.sub(r'\\s+', ' ', text.strip())
    """).strip()


@pytest.fixture
def java_source() -> str:
    return textwrap.dedent("""
        import java.util.List;
        import java.util.ArrayList;

        /** Binary search implementation. */
        public class BinarySearch {

            /** Find index of target in sorted list. */
            public int search(List<Integer> list, int target) {
                int lo = 0, hi = list.size() - 1;
                while (lo <= hi) {
                    int mid = (lo + hi) / 2;
                    if (list.get(mid) == target) return mid;
                    else if (list.get(mid) < target) lo = mid + 1;
                    else hi = mid - 1;
                }
                return -1;
            }

            @Override
            @Deprecated
            public String toString() {
                return "BinarySearch";
            }

            public BinarySearch() {}
        }
    """).strip()


@pytest.fixture
def js_source() -> str:
    return textwrap.dedent("""
        import fs from 'fs';
        import { readFile } from 'fs/promises';
        import * as path from 'path';

        /** File utility class. */
        class FileUtils {
            constructor(basePath) {
                this.basePath = basePath;
            }

            /** Read a file asynchronously. */
            async read(filename) {
                return readFile(path.join(this.basePath, filename), 'utf8');
            }
        }

        /** Copy a file synchronously. */
        function copyFile(src, dest) {
            fs.copyFileSync(src, dest);
        }

        const double = (x) => x * 2;
    """).strip()


# ─── Filesystem fixtures ──────────────────────────────────────────────────────

@pytest.fixture
def tmp_repo(tmp_path):
    """Return a helper that writes a synthetic repo under tmp_path."""
    def _make(structure: dict[str, str]) -> Path:
        for rel, content in structure.items():
            p = tmp_path / rel
            p.parent.mkdir(parents=True, exist_ok=True)
            p.write_text(content)
        return tmp_path
    return _make


@pytest.fixture
def python_repo_with_tests(tmp_repo):
    """A minimal Python repo with all four test signals firing."""
    return tmp_repo({
        "src/app.py": "def run(): pass\n",
        "tests/test_app.py": "def test_run(): pass\n",
        "pytest.ini": "[pytest]\ntestpaths = tests\n",
        ".github/workflows/ci.yml": "jobs:\n  test:\n    steps:\n      - run: pytest\n",
    })


@pytest.fixture
def python_repo_no_tests(tmp_repo):
    """A minimal Python repo with no test infrastructure."""
    return tmp_repo({
        "src/app.py": "def run(): pass\n",
        "README.md": "# My project\n",
    })