"""
core/detection.py

Detects test suite presence in a cloned repository using static heuristics.
No code execution required — this is a purely structural analysis.

Detection strategy (in order of confidence):
1. Test directory names
2. Test file naming patterns
3. Test runner config files
4. CI workflow files that invoke test runners
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
    "__tests__",       # JS/TS
    "spec",            # Ruby / JS
    "specs",
    "unit_tests",
    "integration_tests",
    "functional_tests",
    "e2e",
}

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
}

# Test runner config files (presence = test infrastructure exists)
TEST_CONFIG_FILES = {
    "Python": ["pytest.ini", "setup.cfg", "tox.ini", "pyproject.toml", ".pytest.ini"],
    "Java": ["pom.xml", "build.gradle", "build.gradle.kts"],
    "JavaScript": ["jest.config.js", "jest.config.ts", "jest.config.json",
                   "karma.conf.js", "mocha.opts", ".mocharc.js", ".mocharc.yml"],
}

# Keywords in CI files that indicate test execution
CI_TEST_KEYWORDS = [
    "pytest", "unittest", "nose", "tox",     # Python
    "mvn test", "gradle test", "junit",       # Java
    "jest", "mocha", "karma", "jasmine",      # JS
    "npm test", "yarn test",                  # JS generic
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
    config_files = TEST_CONFIG_FILES.get(language, [])

    for dirpath, dirnames, filenames in os.walk(root):
        # Skip hidden directories and common non-source dirs
        dirnames[:] = [
            d for d in dirnames
            if d not in {"node_modules", "__pycache__", ".git", "venv", ".venv", "env"}
            and (not d.startswith(".") or d == ".github")
        ]

        rel_dir = Path(dirpath).relative_to(root)
        dir_name = Path(dirpath).name.lower()

        # Signal 1: test directory
        if dir_name in TEST_DIRS:
            report.has_test_directory = True
            report.test_directories_found.append(str(rel_dir))

        for filename in filenames:
            # Signal 2: test file naming patterns
            if any(p.match(filename) for p in file_patterns):
                report.has_test_files = True
                report.test_file_count += 1

            # Signal 3: test config files
            if filename in config_files:
                # For pyproject.toml / setup.cfg, verify they reference pytest/unittest
                if filename in ("pyproject.toml", "setup.cfg", "pom.xml", "build.gradle"):
                    try:
                        content = Path(dirpath, filename).read_text(errors="ignore")
                        if any(kw in content for kw in ["pytest", "unittest", "testng", "junit", "jest"]):
                            report.has_test_config = True
                            report.test_config_files_found.append(filename)
                    except Exception:
                        pass
                else:
                    report.has_test_config = True
                    report.test_config_files_found.append(filename)

            # Signal 4: CI workflow files
            if filename.endswith((".yml", ".yaml")) and ".github" in str(dirpath):
                try:
                    content = Path(dirpath, filename).read_text(errors="ignore").lower()
                    if any(kw in content for kw in CI_TEST_KEYWORDS):
                        report.has_ci_test_invocation = True
                except Exception:
                    pass

    logger.debug(
        f"Test detection for {repo_path}: present={report.is_present}, "
        f"confidence={report.confidence}, test_files={report.test_file_count}"
    )
    return report