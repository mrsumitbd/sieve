"""
tests/test_detection.py

Integration tests for detect_test_suite.
Uses real temporary directories — no network or GitHub access.
"""

import pytest

from sieve.core.detection import detect_test_suite


class TestPythonDetection:
    def test_all_signals_present(self, python_repo_with_tests):
        report = detect_test_suite(str(python_repo_with_tests), "Python")
        assert report.is_present is True
        assert report.confidence in ("high", "medium")

    def test_no_tests_detected(self, python_repo_no_tests):
        report = detect_test_suite(str(python_repo_no_tests), "Python")
        assert report.is_present is False

    def test_test_dir_signal(self, tmp_repo):
        repo = tmp_repo({
            "src/app.py": "def run(): pass\n",
            "tests/test_app.py": "import pytest\ndef test_run(): pass\n",
        })
        report = detect_test_suite(str(repo), "Python")
        assert report.is_present is True

    def test_pytest_config_signal(self, tmp_repo):
        repo = tmp_repo({
            "src/app.py": "def run(): pass\n",
            "tests/test_app.py": "def test_run(): pass\n",
            "pytest.ini": "[pytest]\ntestpaths = tests\n",
        })
        report = detect_test_suite(str(repo), "Python")
        assert report.is_present is True

    def test_ci_workflow_signal(self, tmp_repo):
        """CI workflows in .github/workflows/ must be visible (not excluded)."""
        repo = tmp_repo({
            "src/app.py": "def run(): pass\n",
            "tests/test_app.py": "def test_run(): pass\n",
            ".github/workflows/ci.yml": "jobs:\n  test:\n    steps:\n      - run: pytest\n",
        })
        report = detect_test_suite(str(repo), "Python")
        assert report.is_present is True
        # CI signal should be detected
        assert report.has_ci_test_invocation is True

    def test_config_only_does_not_satisfy_threshold(self, tmp_repo):
        """pytest.ini alone (1 signal) should not meet the ≥2 threshold."""
        repo = tmp_repo({
            "src/app.py": "def run(): pass\n",
            "pytest.ini": "[pytest]\ntestpaths = tests\n",
        })
        report = detect_test_suite(str(repo), "Python")
        assert report.is_present is False

    def test_report_to_dict(self, python_repo_with_tests):
        report = detect_test_suite(str(python_repo_with_tests), "Python")
        d = report.to_dict()
        assert "test_suite_present" in d
        assert "confidence" in d
        assert "has_test_directory" in d
        assert "has_ci_test_invocation" in d


class TestJavaDetection:
    def test_java_test_files_detected(self, tmp_repo):
        repo = tmp_repo({
            "src/main/java/App.java": "public class App {}",
            "src/test/java/AppTest.java": "import org.junit.*;\npublic class AppTest {}",
            "build.gradle": "dependencies { testImplementation 'junit:junit:4.13' }",
        })
        report = detect_test_suite(str(repo), "Java")
        assert report.is_present is True

    def test_no_java_tests(self, tmp_repo):
        repo = tmp_repo({
            "src/main/java/App.java": "public class App {}",
        })
        report = detect_test_suite(str(repo), "Java")
        assert report.is_present is False


class TestJavaScriptDetection:
    def test_jest_config_and_test_files_detected(self, tmp_repo):
        repo = tmp_repo({
            "src/app.js": "module.exports = {};",
            "__tests__/app.test.js": "test('a', () => {});",
            "jest.config.js": "module.exports = { testEnvironment: 'node' };",
        })
        report = detect_test_suite(str(repo), "JavaScript")
        assert report.is_present is True

    def test_no_js_tests(self, tmp_repo):
        repo = tmp_repo({
            "src/app.js": "module.exports = {};",
        })
        report = detect_test_suite(str(repo), "JavaScript")
        assert report.is_present is False