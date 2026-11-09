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

    def test_unreadable_config_file_handled_gracefully(self, tmp_repo):
        """Exception when reading pyproject.toml is silently skipped."""
        from unittest.mock import patch
        repo = tmp_repo({
            "src/app.py": "def foo(): pass",
            "pyproject.toml": "[tool.pytest.ini_options]\n",
        })
        with patch("sieve.core.detection.Path.read_text",
                   side_effect=PermissionError("cannot read")):
            # Should not raise
            report = detect_test_suite(str(repo), "Python")
        assert report is not None

    def test_unreadable_ci_yml_handled_gracefully(self, tmp_repo):
        """Exception when reading CI YAML is silently skipped."""
        from unittest.mock import patch
        repo = tmp_repo({
            "src/app.py": "def foo(): pass",
            ".github/workflows/ci.yml": "name: CI\njobs:\n  test:\n    run: pytest\n",
        })
        with patch("sieve.core.detection.Path.read_text",
                   side_effect=PermissionError("cannot read")):
            report = detect_test_suite(str(repo), "Python")
        assert report is not None

# ---------------------------------------------------------------------------
# False positives and path bugs (fix/test-suite-detection)
# ---------------------------------------------------------------------------

PYTEST_IN_DEPS = (
    "[project]\nname = 'x'\n"
    "[project.optional-dependencies]\ndev = ['pytest>=7']\n"
)


class TestTestDirectoryNeedsRealTests:
    def test_empty_of_tests_dir_does_not_count(self, tmp_repo):
        repo = tmp_repo({
            "src/app.py": "def run(): pass\n",
            "tests/data/sample.json": "{}",
            "tests/fixtures.txt": "x",
        })
        report = detect_test_suite(str(repo), "Python")
        assert report.has_test_directory is False
        assert report.test_directories_found == []

    def test_dir_with_test_files_counts(self, tmp_repo):
        repo = tmp_repo({"tests/test_a.py": "def test_a(): pass\n"})
        report = detect_test_suite(str(repo), "Python")
        assert report.has_test_directory is True
        assert report.test_directories_found == ["tests"]

    def test_nested_test_files_count_for_outer_test_dir(self, tmp_repo):
        repo = tmp_repo({"tests/unit/core/test_a.py": "def test_a(): pass\n"})
        report = detect_test_suite(str(repo), "Python")
        assert report.has_test_directory is True
        assert "tests" in report.test_directories_found

    def test_nested_package_test_dir(self, tmp_repo):
        repo = tmp_repo({
            "src/pkg/app.py": "x = 1\n",
            "src/pkg/tests/test_app.py": "def test_x(): pass\n",
        })
        report = detect_test_suite(str(repo), "Python")
        assert report.has_test_directory is True
        assert report.test_file_count == 1

    def test_data_dir_plus_pytest_dependency_is_not_a_suite(self, tmp_repo):
        """The reported false positive: two weak signals together."""
        repo = tmp_repo({
            "src/app.py": "def run(): pass\n",
            "testing/data/blob.bin": "x",
            "pyproject.toml": PYTEST_IN_DEPS,
        })
        report = detect_test_suite(str(repo), "Python")
        assert report.has_test_directory is False
        assert report.has_test_config is False
        assert report.is_present is False

    def test_javascript_empty_test_dir_does_not_count(self, tmp_repo):
        repo = tmp_repo({
            "src/app.js": "module.exports = {};",
            "test/fixtures/data.json": "{}",
            "jest.config.js": "module.exports = {};",
        })
        report = detect_test_suite(str(repo), "JavaScript")
        assert report.has_test_directory is False
        assert report.is_present is False


class TestPytestConfigMustBeReal:
    def test_pyproject_mentioning_pytest_only_as_dependency(self, tmp_repo):
        repo = tmp_repo({"pyproject.toml": PYTEST_IN_DEPS})
        report = detect_test_suite(str(repo), "Python")
        assert report.has_test_config is False
        assert report.test_config_files_found == []

    def test_pyproject_with_pytest_section(self, tmp_repo):
        repo = tmp_repo({
            "pyproject.toml": "[tool.pytest.ini_options]\ntestpaths = ['tests']\n",
        })
        report = detect_test_suite(str(repo), "Python")
        assert report.has_test_config is True
        assert report.test_config_files_found == ["pyproject.toml"]

    def test_setup_cfg_dependency_only(self, tmp_repo):
        repo = tmp_repo({
            "setup.cfg": "[options.extras_require]\ndev =\n    pytest\n",
        })
        assert detect_test_suite(str(repo), "Python").has_test_config is False

    def test_setup_cfg_tool_pytest_section(self, tmp_repo):
        repo = tmp_repo({"setup.cfg": "[tool:pytest]\naddopts = -q\n"})
        assert detect_test_suite(str(repo), "Python").has_test_config is True

    def test_setup_cfg_pytest_section(self, tmp_repo):
        repo = tmp_repo({"setup.cfg": "[pytest]\naddopts = -q\n"})
        assert detect_test_suite(str(repo), "Python").has_test_config is True

    def test_tox_ini_without_pytest_section(self, tmp_repo):
        repo = tmp_repo({"tox.ini": "[tox]\nenvlist = py311\n[flake8]\nmax-line-length=99\n"})
        assert detect_test_suite(str(repo), "Python").has_test_config is False

    def test_tox_ini_with_pytest_section(self, tmp_repo):
        repo = tmp_repo({"tox.ini": "[pytest]\ntestpaths = tests\n"})
        assert detect_test_suite(str(repo), "Python").has_test_config is True

    def test_pytest_ini_without_pytest_section_is_not_config(self, tmp_repo):
        repo = tmp_repo({"pytest.ini": "# nothing here\n"})
        assert detect_test_suite(str(repo), "Python").has_test_config is False

    def test_pytest_ini_with_section(self, tmp_repo):
        repo = tmp_repo({"pytest.ini": "[pytest]\ntestpaths = tests\n"})
        assert detect_test_suite(str(repo), "Python").has_test_config is True

    def test_conftest_py_counts(self, tmp_repo):
        repo = tmp_repo({"conftest.py": "import pytest\n"})
        report = detect_test_suite(str(repo), "Python")
        assert report.has_test_config is True
        assert "conftest.py" in report.test_config_files_found

    def test_nested_conftest_py_counts(self, tmp_repo):
        repo = tmp_repo({"tests/conftest.py": "import pytest\n"})
        assert detect_test_suite(str(repo), "Python").has_test_config is True

    def test_conftest_py_not_config_for_other_languages(self, tmp_repo):
        repo = tmp_repo({"conftest.py": "import pytest\n"})
        assert detect_test_suite(str(repo), "Java").has_test_config is False

    def test_java_pom_with_junit_still_counts(self, tmp_repo):
        repo = tmp_repo({"pom.xml": "<dependency><artifactId>junit</artifactId></dependency>"})
        assert detect_test_suite(str(repo), "Java").has_test_config is True


class TestVendoredDirectoriesSkipped:
    @pytest.mark.parametrize("vendor", [
        "venv", ".venv", "env", "node_modules", "__pycache__",
        "site-packages", "vendor", "vendored", "third_party", "third-party",
        "external", "build", "dist", ".tox", ".eggs", ".mypy_cache", ".pytest_cache",
    ])
    def test_vendored_tests_are_not_counted(self, tmp_repo, vendor):
        repo = tmp_repo({
            "src/app.py": "def run(): pass\n",
            f"{vendor}/lib/tests/test_lib.py": "def test_x(): pass\n",
            f"{vendor}/lib/test_other.py": "def test_y(): pass\n",
        })
        report = detect_test_suite(str(repo), "Python")
        assert report.test_file_count == 0
        assert report.has_test_files is False
        assert report.has_test_directory is False
        assert report.is_present is False

    def test_skip_list_is_one_shared_constant(self):
        from sieve.core import detection
        assert isinstance(detection.SKIP_DIRS, frozenset)
        for name in ("site-packages", "vendor", "vendored", "third_party",
                     "third-party", "external", "build", "dist", ".tox",
                     ".eggs", ".mypy_cache", ".pytest_cache", "node_modules"):
            assert name in detection.SKIP_DIRS

    def test_real_tests_alongside_vendor_counted_exactly(self, tmp_repo):
        repo = tmp_repo({
            "tests/test_a.py": "def test_a(): pass\n",
            "vendor/dep/test_dep.py": "def test_d(): pass\n",
            "third_party/x/test_x.py": "def test_x(): pass\n",
        })
        report = detect_test_suite(str(repo), "Python")
        assert report.test_file_count == 1

    def test_github_dir_still_visited(self, tmp_repo):
        repo = tmp_repo({".github/workflows/ci.yml": "run: pytest\n"})
        assert detect_test_suite(str(repo), "Python").has_ci_test_invocation is True


class TestCiDetectionUsesRelativePath:
    def test_clone_path_containing_dot_github_is_not_ci(self, tmp_path):
        repo = tmp_path / ".github" / "clones" / "repo"
        (repo / "config").mkdir(parents=True)
        (repo / "config" / "settings.yml").write_text("runner: pytest\n")
        (repo / "app.py").write_text("x = 1\n")
        report = detect_test_suite(str(repo), "Python")
        assert report.has_ci_test_invocation is False

    def test_clone_path_containing_dot_github_real_workflow_still_found(self, tmp_path):
        repo = tmp_path / ".github" / "clones" / "repo"
        (repo / ".github" / "workflows").mkdir(parents=True)
        (repo / ".github" / "workflows" / "ci.yml").write_text("run: pytest\n")
        assert detect_test_suite(str(repo), "Python").has_ci_test_invocation is True

    def test_yaml_outside_github_is_not_ci(self, tmp_repo):
        repo = tmp_repo({"docs/example.yaml": "command: pytest\n"})
        assert detect_test_suite(str(repo), "Python").has_ci_test_invocation is False

    def test_nested_github_dir_in_monorepo_package(self, tmp_repo):
        repo = tmp_repo({"pkg/.github/workflows/ci.yml": "run: npm test\n"})
        assert detect_test_suite(str(repo), "JavaScript").has_ci_test_invocation is True

    def test_workflow_without_test_command_is_not_ci_test(self, tmp_repo):
        repo = tmp_repo({".github/workflows/release.yml": "run: twine upload dist/*\n"})
        assert detect_test_suite(str(repo), "Python").has_ci_test_invocation is False

    def test_dot_github_clone_path_with_only_tests_dir_and_yaml(self, tmp_path):
        """Combined effect: bogus CI signal + empty tests dir must not mark a repo."""
        repo = tmp_path / ".github" / "work" / "repo"
        (repo / "tests" / "data").mkdir(parents=True)
        (repo / "tests" / "data" / "x.json").write_text("{}")
        (repo / "cfg.yml").write_text("tool: pytest\n")
        assert detect_test_suite(str(repo), "Python").is_present is False


class TestGenuineSuitesStillDetected:
    def test_python_full_layout(self, tmp_repo):
        repo = tmp_repo({
            "src/pkg/__init__.py": "",
            "src/pkg/core.py": "def f(): pass\n",
            "tests/__init__.py": "",
            "tests/test_core.py": "def test_f(): pass\n",
            "tests/unit/test_more.py": "def test_g(): pass\n",
            "pyproject.toml": "[tool.pytest.ini_options]\ntestpaths = ['tests']\n",
        })
        report = detect_test_suite(str(repo), "Python")
        assert report.is_present is True
        assert report.confidence == "high"
        assert report.test_file_count == 2
        assert report.has_test_directory and report.has_test_files and report.has_test_config

    def test_python_conftest_and_tests(self, tmp_repo):
        repo = tmp_repo({
            "app.py": "x = 1\n",
            "test/test_app.py": "def test_x(): pass\n",
            "conftest.py": "",
        })
        assert detect_test_suite(str(repo), "Python").is_present is True

    def test_python_suffix_style_tests(self, tmp_repo):
        repo = tmp_repo({
            "app.py": "x = 1\n",
            "tests/app_test.py": "def test_x(): pass\n",
        })
        assert detect_test_suite(str(repo), "Python").is_present is True

    def test_ci_workflow_that_invokes_tests(self, tmp_repo):
        repo = tmp_repo({
            "app.py": "x = 1\n",
            "tests/test_app.py": "def test_x(): pass\n",
            ".github/workflows/ci.yml": "steps:\n  - run: tox\n",
        })
        report = detect_test_suite(str(repo), "Python")
        assert report.has_ci_test_invocation is True
        assert report.is_present is True

    def test_ci_plus_test_files_without_directory_name(self, tmp_repo):
        repo = tmp_repo({
            "app.py": "x = 1\n",
            "checks/test_app.py": "def test_x(): pass\n",
            ".github/workflows/ci.yaml": "steps:\n  - run: pytest\n",
        })
        report = detect_test_suite(str(repo), "Python")
        assert report.has_test_directory is False
        assert report.is_present is True

    def test_javascript_layout(self, tmp_repo):
        repo = tmp_repo({
            "src/index.js": "module.exports = {};",
            "test/index.test.js": "test('a', () => {});",
            "src/util.spec.ts": "it('b', () => {});",
            "jest.config.js": "module.exports = {};",
            ".github/workflows/node.yml": "steps:\n  - run: npm test\n",
        })
        report = detect_test_suite(str(repo), "JavaScript")
        assert report.is_present is True
        assert report.confidence == "high"
        assert report.test_file_count == 2

    def test_java_maven_layout(self, tmp_repo):
        repo = tmp_repo({
            "pom.xml": "<dependencies><dependency>junit</dependency></dependencies>",
            "src/main/java/com/x/App.java": "public class App {}",
            "src/test/java/com/x/AppTest.java": "public class AppTest {}",
            "src/test/java/com/x/AppIT.java": "public class AppIT {}",
        })
        report = detect_test_suite(str(repo), "Java")
        assert report.is_present is True
        assert report.confidence == "high"
        assert report.test_file_count == 2

    def test_cpp_layout(self, tmp_repo):
        repo = tmp_repo({
            "CMakeLists.txt": "project(x)",
            "src/a.cpp": "int a(){return 1;}",
            "tests/a_test.cpp": "int main(){}",
        })
        assert detect_test_suite(str(repo), "C++").is_present is True

    def test_monorepo_nested_layout(self, tmp_repo):
        repo = tmp_repo({
            "packages/a/src/a.py": "x = 1\n",
            "packages/a/tests/test_a.py": "def test_a(): pass\n",
            "packages/a/pytest.ini": "[pytest]\n",
        })
        report = detect_test_suite(str(repo), "Python")
        assert report.is_present is True
        assert report.test_directories_found == ["packages/a/tests"]

    def test_clone_path_containing_dot_github_genuine_repo(self, tmp_path):
        repo = tmp_path / ".github" / "clones" / "repo"
        (repo / "tests").mkdir(parents=True)
        (repo / "tests" / "test_a.py").write_text("def test_a(): pass\n")
        (repo / "pytest.ini").write_text("[pytest]\n")
        report = detect_test_suite(str(repo), "Python")
        assert report.is_present is True
        assert report.has_ci_test_invocation is False


# ---------------------------------------------------------------------------
# Jest-style __tests__ / __test__ layouts: plain source files count
# ---------------------------------------------------------------------------

class TestJestStyleTestDirectories:
    @pytest.mark.parametrize("dirname", ["__tests__", "__test__"])
    @pytest.mark.parametrize("ext", ["js", "jsx", "mjs", "cjs", "ts", "tsx", "mts", "cts"])
    def test_plain_js_ts_file_in_dunder_dir_counts(self, tmp_repo, dirname, ext):
        repo = tmp_repo({
            "src/app.js": "module.exports = {};",
            f"{dirname}/foo.{ext}": "test('a', () => {});",
        })
        report = detect_test_suite(str(repo), "JavaScript")
        assert report.has_test_directory is True
        assert report.test_directories_found == [dirname]

    def test_jest_layout_with_plain_files_and_config_is_present(self, tmp_repo):
        repo = tmp_repo({
            "src/app.js": "module.exports = {};",
            "src/__tests__/foo.js": "test('a', () => {});",
            "src/__tests__/helpers/bar.js": "module.exports = {};",
            "jest.config.js": "module.exports = {};",
        })
        report = detect_test_suite(str(repo), "JavaScript")
        assert report.has_test_directory is True
        assert report.test_directories_found == ["src/__tests__"]
        assert report.is_present is True

    def test_nested_below_dunder_dir_counts(self, tmp_repo):
        repo = tmp_repo({"__tests__/unit/deep/foo.js": "test('a', () => {});"})
        report = detect_test_suite(str(repo), "JavaScript")
        assert report.test_directories_found == ["__tests__"]

    def test_only_non_source_files_does_not_count(self, tmp_repo):
        repo = tmp_repo({
            "src/app.js": "module.exports = {};",
            "__tests__/data.json": "{}",
            "__tests__/__snapshots__/foo.test.js.snap": "exports[`a`] = 1;",
            "__tests__/README.md": "notes",
            "__tests__/fixture.txt": "x",
        })
        report = detect_test_suite(str(repo), "JavaScript")
        assert report.has_test_directory is False
        assert report.test_directories_found == []

    def test_source_file_outside_dunder_dir_is_not_a_test(self, tmp_repo):
        repo = tmp_repo({"src/foo.js": "x", "helpers/bar.js": "y"})
        report = detect_test_suite(str(repo), "JavaScript")
        assert report.has_test_directory is False

    def test_plain_file_in_ordinary_tests_dir_still_does_not_count(self, tmp_repo):
        """The rule is specific to __tests__/__test__, not test/ or tests/."""
        repo = tmp_repo({"test/helper.js": "module.exports = {};"})
        assert detect_test_suite(str(repo), "JavaScript").has_test_directory is False

    @pytest.mark.parametrize("vendor", sorted([
        "node_modules", "vendor", "third_party", "site-packages", "build", "dist",
    ]))
    def test_dunder_dir_under_skip_dir_does_not_count(self, tmp_repo, vendor):
        repo = tmp_repo({
            "src/app.js": "module.exports = {};",
            f"{vendor}/lib/__tests__/foo.js": "test('a', () => {});",
        })
        report = detect_test_suite(str(repo), "JavaScript")
        assert report.has_test_directory is False
        assert report.test_directories_found == []
        assert report.is_present is False

    def test_only_language_source_extensions_count(self, tmp_repo):
        """A .py file in __tests__ means nothing for a JavaScript repo."""
        repo = tmp_repo({"__tests__/foo.py": "x = 1\n"})
        assert detect_test_suite(str(repo), "JavaScript").has_test_directory is False

    def test_other_languages_use_their_own_source_extension(self, tmp_repo):
        repo = tmp_repo({"__tests__/helper.py": "x = 1\n"})
        assert detect_test_suite(str(repo), "Python").has_test_directory is True

    def test_plain_files_in_dunder_dir_are_counted_as_test_files(self, tmp_repo):
        repo = tmp_repo({
            "__tests__/foo.js": "x",
            "__tests__/real.test.js": "test('a', () => {});",
        })
        report = detect_test_suite(str(repo), "JavaScript")
        assert report.test_file_count == 2
        assert report.has_test_files is True

    def test_plain_file_counted_once_when_it_also_matches_a_pattern(self, tmp_repo):
        repo = tmp_repo({"__tests__/real.test.js": "test('a', () => {});"})
        assert detect_test_suite(str(repo), "JavaScript").test_file_count == 1

    def test_non_source_files_in_dunder_dir_not_counted(self, tmp_repo):
        repo = tmp_repo({
            "__tests__/data.json": "{}",
            "__tests__/__snapshots__/a.test.js.snap": "x",
        })
        report = detect_test_suite(str(repo), "JavaScript")
        assert report.test_file_count == 0
        assert report.has_test_files is False

    def test_wrong_language_extension_not_counted_as_test_file(self, tmp_repo):
        repo = tmp_repo({"__tests__/foo.py": "x = 1\n"})
        report = detect_test_suite(str(repo), "JavaScript")
        assert report.test_file_count == 0
        assert report.has_test_files is False

    def test_jest_layout_without_config_is_detected(self, tmp_repo):
        repo = tmp_repo({
            "src/app.js": "module.exports = {};",
            "src/__tests__/foo.js": "test('a', () => {});",
        })
        report = detect_test_suite(str(repo), "JavaScript")
        assert report.has_test_directory is True
        assert report.has_test_files is True
        assert report.has_test_config is False
        assert report.test_file_count == 1
        assert report.is_present is True

    def test_jest_layout_without_config_under_node_modules_is_not_detected(self, tmp_repo):
        repo = tmp_repo({
            "src/app.js": "module.exports = {};",
            "node_modules/pkg/src/__tests__/foo.js": "test('a', () => {});",
        })
        report = detect_test_suite(str(repo), "JavaScript")
        assert report.has_test_directory is False
        assert report.has_test_files is False
        assert report.test_file_count == 0
        assert report.is_present is False

    @pytest.mark.parametrize("vendor", ["vendor", "third_party", "site-packages", "build", "dist"])
    def test_dunder_files_under_skip_dir_not_counted(self, tmp_repo, vendor):
        repo = tmp_repo({f"{vendor}/lib/__tests__/foo.js": "x"})
        report = detect_test_suite(str(repo), "JavaScript")
        assert report.test_file_count == 0
        assert report.has_test_files is False
