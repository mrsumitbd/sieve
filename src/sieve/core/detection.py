"""
core/detection.py

Detects test suite presence in a cloned repository using static heuristics.
No code execution required — this is a purely structural analysis.

Detection strategy (in order of confidence):
1. Test directories (a test-named directory that contains real test files)
2. Test file naming patterns
3. Test runner config (an actual pytest configuration for Python)
4. CI workflow files (under .github/) that invoke test runners

Vendored, generated and virtualenv directories (see SKIP_DIRS) are never walked.
"""

import os
import re
import logging
from pathlib import Path
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)

# High-confidence test directory names
TEST_DIRS = {
    "test", "tests", "test_suite", "testing",
    "__tests__", "__test__",   # JS/TS
    "spec",            # Ruby / JS
    "specs",
    "unit_tests",
    "integration_tests",
    "functional_tests",
    "e2e",
}

# Directories never walked: virtualenvs, caches, build output and vendored or
# third-party code. Tests found in these belong to someone else's project.
SKIP_DIRS = frozenset({
    ".git",
    "venv", ".venv", "env",
    "node_modules", "__pycache__",
    "site-packages",
    "vendor", "vendored", "third_party", "third-party", "external",
    "build", "dist",
    ".tox", ".eggs", ".mypy_cache", ".pytest_cache",
})

# Test file patterns per language
TEST_FILE_PATTERNS = {
    "Python": [
        re.compile(r"^test_.*\.py$"),
        re.compile(r".*_test\.py$"),
    ],
    "Java": [
        re.compile(r".*Test\.java$"),
        re.compile(r".*Tests\.java$"),
        re.compile(r".*TestCase\.java$"),
        re.compile(r".*IT\.java$"),          # Integration tests
    ],
    "JavaScript": [
        re.compile(r".*\.test\.js$"),
        re.compile(r".*\.spec\.js$"),
        re.compile(r".*\.test\.ts$"),
        re.compile(r".*\.spec\.ts$"),
    ],
    "C++": [
        re.compile(r"^test_.*\.(cpp|cc|cxx)$"),
        re.compile(r".*_test\.(cpp|cc|cxx)$"),
        re.compile(r".*[Tt]est\.(cpp|cc|cxx)$"),
        re.compile(r".*[Ss]pec\.(cpp|cc|cxx)$"),
    ],
}

# Directories where every source file is a test (Jest's default layout puts
# plain foo.js files in __tests__/). Such files count as test files for the
# test-directory requirement, has_test_files and test_file_count.
TEST_ONLY_DIRS = frozenset({"__tests__", "__test__"})

# Normal source extensions per language, for files inside TEST_ONLY_DIRS.
SOURCE_EXTENSIONS = {
    "Python": frozenset({".py"}),
    "Java": frozenset({".java"}),
    "JavaScript": frozenset({".js", ".jsx", ".mjs", ".cjs",
                             ".ts", ".tsx", ".mts", ".cts"}),
    "C++": frozenset({".cpp", ".cc", ".cxx", ".h", ".hpp"}),
}

# Test runner config files (presence = test infrastructure exists)
TEST_CONFIG_FILES = {
    "Python": ["pytest.ini", "setup.cfg", "tox.ini", "pyproject.toml", ".pytest.ini",
               "conftest.py"],
    "Java": ["pom.xml", "build.gradle", "build.gradle.kts"],
    "JavaScript": ["jest.config.js", "jest.config.ts", "jest.config.json",
                   "karma.conf.js", "mocha.opts", ".mocharc.js", ".mocharc.yml"],
    "C++": ["CMakeLists.txt", "CTestTestfile.cmake", "googletest.cmake",
            "Makefile", "meson.build"],
}

# A Python file only counts as pytest configuration when it has the real section,
# not merely the word "pytest" (usually a dev dependency).
_PYTEST_INI_SECTION = re.compile(r"^\s*\[\s*(?:tool:pytest|pytest)\s*\]", re.MULTILINE)
_PYTEST_TOML_SECTION = re.compile(r"^\s*\[\s*tool\.pytest(?:\.ini_options)?\s*\]", re.MULTILINE)
_PYTEST_INI_FILES = ("pytest.ini", ".pytest.ini", "setup.cfg", "tox.ini")

# Keywords in CI files that indicate test execution
CI_TEST_KEYWORDS = [
    "pytest", "unittest", "nose", "tox",         # Python
    "mvn test", "gradle test", "junit",            # Java
    "jest", "mocha", "karma", "jasmine",           # JS
    "npm test", "yarn test",                       # JS generic
    "ctest", "gtest", "googletest", "catch2",      # C++
    "run tests", "run_tests",
]


@dataclass
class TestSuiteReport:
    has_test_directory: bool = False
    has_test_files: bool = False
    has_test_config: bool = False
    has_ci_test_invocation: bool = False
    test_file_count: int = 0
    test_directories_found: list[str] = field(default_factory=list)
    test_config_files_found: list[str] = field(default_factory=list)

    @property
    def is_present(self) -> bool:
        """True if at least two signals fire — reduces false positives."""
        signals = [
            self.has_test_directory,
            self.has_test_files,
            self.has_test_config,
            self.has_ci_test_invocation,
        ]
        return sum(signals) >= 2

    @property
    def confidence(self) -> str:
        signals = sum([
            self.has_test_directory,
            self.has_test_files,
            self.has_test_config,
            self.has_ci_test_invocation,
        ])
        if signals >= 3:
            return "high"
        elif signals == 2:
            return "medium"
        elif signals == 1:
            return "low"
        return "none"

    def to_dict(self) -> dict:
        return {
            "test_suite_present": self.is_present,
            "confidence": self.confidence,
            "has_test_directory": self.has_test_directory,
            "has_test_files": self.has_test_files,
            "has_test_config": self.has_test_config,
            "has_ci_test_invocation": self.has_ci_test_invocation,
            "test_file_count": self.test_file_count,
            "test_directories_found": self.test_directories_found,
            "test_config_files_found": self.test_config_files_found,
        }


def _is_test_config(path: Path, language: str) -> bool:
    """True if `path` (already matched by name) is a real test runner config."""
    name = path.name
    if name == "conftest.py":
        return True
    if name not in ("pyproject.toml", "setup.cfg", "pom.xml", "build.gradle") and \
            name not in _PYTEST_INI_FILES:
        return True  # JS/C++ config files: presence is enough

    try:
        content = path.read_text(errors="ignore")
    except Exception:
        return False

    if name == "pyproject.toml":
        return bool(_PYTEST_TOML_SECTION.search(content))
    if name in _PYTEST_INI_FILES:
        return bool(_PYTEST_INI_SECTION.search(content))
    # pom.xml / build.gradle
    return any(kw in content for kw in ("pytest", "unittest", "testng", "junit", "jest"))


def detect_test_suite(repo_path: str, language: str) -> TestSuiteReport:
    """
    Statically analyze a cloned repo for test suite presence.

    Args:
        repo_path: Absolute path to the cloned repository
        language: Programming language (affects file pattern matching)

    Returns:
        TestSuiteReport with all detection signals
    """
    root = Path(repo_path)
    report = TestSuiteReport()
    file_patterns = TEST_FILE_PATTERNS.get(language, [])
    source_exts = SOURCE_EXTENSIONS.get(language, frozenset())
    config_files = TEST_CONFIG_FILES.get(language, [])

    test_dirs: set[str] = set()

    for dirpath, dirnames, filenames in os.walk(root):
        # Skip vendored/generated dirs and hidden dirs (except .github)
        dirnames[:] = [
            d for d in dirnames
            if d not in SKIP_DIRS and (not d.startswith(".") or d == ".github")
        ]

        rel_dir = Path(dirpath).relative_to(root)

        in_test_only_dir = any(part.lower() in TEST_ONLY_DIRS for part in rel_dir.parts)

        for filename in filenames:
            is_test_file = any(p.match(filename) for p in file_patterns) or (
                in_test_only_dir and Path(filename).suffix in source_exts
            )

            if is_test_file:
                # Signal 2: test files
                report.has_test_files = True
                report.test_file_count += 1

                # Signal 1: a test-named directory only counts when it holds a
                # real test file (data/fixture folders do not).
                for i, part in enumerate(rel_dir.parts):
                    if part.lower() in TEST_DIRS:
                        test_dirs.add(str(Path(*rel_dir.parts[: i + 1])))

            # Signal 3: test config files
            if filename in config_files and _is_test_config(Path(dirpath, filename), language):
                report.has_test_config = True
                report.test_config_files_found.append(filename)

            # Signal 4: CI workflow files (relative path, so the clone location
            # cannot make unrelated YAML look like CI)
            if filename.endswith((".yml", ".yaml")) and ".github" in rel_dir.parts:
                try:
                    content = Path(dirpath, filename).read_text(errors="ignore").lower()
                    if any(kw in content for kw in CI_TEST_KEYWORDS):
                        report.has_ci_test_invocation = True
                except Exception:
                    pass

    report.test_directories_found = sorted(test_dirs)
    report.has_test_directory = bool(test_dirs)

    logger.debug(
        f"Test detection for {repo_path}: present={report.is_present}, "
        f"confidence={report.confidence}, test_files={report.test_file_count}"
    )
    return report