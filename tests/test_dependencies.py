"""
tests/test_dependencies.py

Unit tests for sieve.core.dependencies — manifest parsing across languages.
"""

import json
import pytest
from pathlib import Path
from sieve.core.dependencies import (
    parse_dependencies,
    _parse_requirements_txt,
    _parse_pyproject_toml,
    _parse_pyproject_toml_regex,
    _parse_python_deps,
    _parse_js_deps,
    _parse_java_deps,
    _parse_cpp_deps,
)


def _write(tmp_path, rel_path, content):
    p = tmp_path / rel_path
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(content, encoding="utf-8")
    return p


class TestParseRequirementsTxt:
    def test_simple_package(self, tmp_path):
        p = _write(tmp_path, "requirements.txt", "requests\n")
        deps = _parse_requirements_txt(p)
        assert any(d["name"] == "requests" for d in deps)

    def test_versioned_package(self, tmp_path):
        p = _write(tmp_path, "requirements.txt", "flask>=2.0.0\n")
        deps = _parse_requirements_txt(p)
        d = next(d for d in deps if d["name"] == "flask")
        assert d["version"] is not None and "2.0.0" in d["version"]

    def test_comments_ignored(self, tmp_path):
        p = _write(tmp_path, "requirements.txt", "# comment\nrequests\n")
        deps = _parse_requirements_txt(p)
        assert not any(d["name"].startswith("#") for d in deps)

    def test_flags_ignored(self, tmp_path):
        p = _write(tmp_path, "requirements.txt", "-r other.txt\nrequests\n")
        deps = _parse_requirements_txt(p)
        assert not any(d["name"].startswith("-") for d in deps)

    def test_multiple_packages(self, tmp_path):
        p = _write(tmp_path, "requirements.txt", "requests>=2.28\nnumpy==1.24.0\npandas\n")
        deps = _parse_requirements_txt(p)
        names = [d["name"] for d in deps]
        assert "requests" in names and "numpy" in names and "pandas" in names

    def test_all_kind_is_main(self, tmp_path):
        p = _write(tmp_path, "requirements.txt", "requests\nnumpy\n")
        deps = _parse_requirements_txt(p)
        assert all(d["kind"] == "main" for d in deps)

    def test_empty_file(self, tmp_path):
        p = _write(tmp_path, "requirements.txt", "")
        assert _parse_requirements_txt(p) == []


class TestParsePyprojectToml:
    """Covers PEP 621 `[project]`, Poetry, and the regex-fallback path --
    previously entirely untested (0 of ~61 lines)."""

    def test_pep621_dependencies_parsed(self, tmp_path):
        content = """
[project]
name = "myapp"
dependencies = [
    "requests>=2.28",
    "numpy",
]
"""
        p = _write(tmp_path, "pyproject.toml", content)
        deps = _parse_pyproject_toml(p)
        names = [d["name"] for d in deps]
        assert "requests" in names and "numpy" in names

    def test_pep621_dependency_kind_is_main(self, tmp_path):
        content = '[project]\ndependencies = ["requests"]\n'
        p = _write(tmp_path, "pyproject.toml", content)
        deps = _parse_pyproject_toml(p)
        assert all(d["kind"] == "main" for d in deps)

    def test_pep621_version_captured(self, tmp_path):
        content = '[project]\ndependencies = ["flask>=2.0.0"]\n'
        p = _write(tmp_path, "pyproject.toml", content)
        deps = _parse_pyproject_toml(p)
        d = next(d for d in deps if d["name"] == "flask")
        assert d["version"] is not None and "2.0.0" in d["version"]

    def test_pep621_extras_stripped_from_name(self, tmp_path):
        content = '[project]\ndependencies = ["requests[security]>=2.0"]\n'
        p = _write(tmp_path, "pyproject.toml", content)
        deps = _parse_pyproject_toml(p)
        names = [d["name"] for d in deps]
        assert "requests" in names
        assert not any("[" in n for n in names)

    def test_pep621_optional_dependencies_kind(self, tmp_path):
        content = """
[project]
dependencies = ["requests"]

[project.optional-dependencies]
dev = ["pytest", "black"]
"""
        p = _write(tmp_path, "pyproject.toml", content)
        deps = _parse_pyproject_toml(p)
        opt = {d["name"]: d["kind"] for d in deps if d["name"] in ("pytest", "black")}
        assert opt == {"pytest": "optional", "black": "optional"}

    def test_pep621_multiple_optional_groups_all_captured(self, tmp_path):
        content = """
[project.optional-dependencies]
dev = ["pytest"]
docs = ["sphinx"]
"""
        p = _write(tmp_path, "pyproject.toml", content)
        deps = _parse_pyproject_toml(p)
        names = [d["name"] for d in deps]
        assert "pytest" in names and "sphinx" in names

    def test_poetry_string_version_captured(self, tmp_path):
        content = """
[tool.poetry.dependencies]
python = "^3.11"
requests = "^2.28.0"
"""
        p = _write(tmp_path, "pyproject.toml", content)
        deps = _parse_pyproject_toml(p)
        d = next(d for d in deps if d["name"] == "requests")
        assert d["version"] == "^2.28.0" and d["kind"] == "main"

    def test_poetry_python_key_skipped(self, tmp_path):
        content = """
[tool.poetry.dependencies]
python = "^3.11"
requests = "^2.28.0"
"""
        p = _write(tmp_path, "pyproject.toml", content)
        deps = _parse_pyproject_toml(p)
        assert "python" not in [d["name"] for d in deps]

    def test_poetry_dict_form_version_captured(self, tmp_path):
        content = """
[tool.poetry.dependencies]
numpy = { version = "^1.24", optional = true }
"""
        p = _write(tmp_path, "pyproject.toml", content)
        deps = _parse_pyproject_toml(p)
        d = next(d for d in deps if d["name"] == "numpy")
        assert d["version"] == "^1.24"

    def test_poetry_dict_form_without_version_is_none(self, tmp_path):
        content = """
[tool.poetry.dependencies]
mylocalpkg = { path = "../mylocalpkg" }
"""
        p = _write(tmp_path, "pyproject.toml", content)
        deps = _parse_pyproject_toml(p)
        d = next(d for d in deps if d["name"] == "mylocalpkg")
        assert d["version"] is None

    def test_poetry_dev_dependencies_kind(self, tmp_path):
        content = """
[tool.poetry.dev-dependencies]
pytest = "^8.0"
"""
        p = _write(tmp_path, "pyproject.toml", content)
        deps = _parse_pyproject_toml(p)
        d = next(d for d in deps if d["name"] == "pytest")
        assert d["kind"] == "dev"

    def test_empty_pyproject_returns_empty(self, tmp_path):
        p = _write(tmp_path, "pyproject.toml", "[project]\nname = \"myapp\"\n")
        assert _parse_pyproject_toml(p) == []

    def test_malformed_toml_falls_back_to_regex(self, tmp_path):
        # Missing closing bracket on the table header makes this invalid TOML,
        # forcing tomllib.loads() to raise and _parse_pyproject_toml() to hit
        # its `except Exception: return _parse_pyproject_toml_regex(path)`.
        content = '[project\ndependencies = ["requests>=2.0", "flask"]\n'
        p = _write(tmp_path, "pyproject.toml", content)
        deps = _parse_pyproject_toml(p)
        names = [d["name"] for d in deps]
        assert "requests" in names and "flask" in names

    def test_regex_fallback_called_directly(self, tmp_path):
        content = 'dependencies = ["click>=8.0", "rich"]\n'
        p = _write(tmp_path, "pyproject.toml", content)
        deps = _parse_pyproject_toml_regex(p)
        names = [d["name"] for d in deps]
        assert "click" in names and "rich" in names

    def test_regex_fallback_no_dependencies_block_returns_empty(self, tmp_path):
        p = _write(tmp_path, "pyproject.toml", "[project]\nname = \"myapp\"\n")
        assert _parse_pyproject_toml_regex(p) == []

    def test_regex_fallback_strips_extras(self, tmp_path):
        content = 'dependencies = ["requests[security]>=2.0"]\n'
        p = _write(tmp_path, "pyproject.toml", content)
        deps = _parse_pyproject_toml_regex(p)
        names = [d["name"] for d in deps]
        assert "requests" in names
        assert not any("[" in n for n in names)

    def test_regex_fallback_extras_do_not_truncate_later_entries(self, tmp_path):
        # Regression test: a naive "stop at the first ']'" regex would treat
        # the ']' closing "[security]" as the end of the whole list, silently
        # dropping every entry that comes after it.
        content = 'dependencies = ["requests[security]>=2.0", "flask", "numpy"]\n'
        p = _write(tmp_path, "pyproject.toml", content)
        deps = _parse_pyproject_toml_regex(p)
        names = [d["name"] for d in deps]
        assert names == ["requests", "flask", "numpy"]

    def test_regex_fallback_unpinned_dependency_does_not_crash(self, tmp_path):
        # Regression test: version-group can legitimately be absent (no
        # version pin at all); this must not raise AttributeError.
        content = 'dependencies = ["rich"]\n'
        p = _write(tmp_path, "pyproject.toml", content)
        deps = _parse_pyproject_toml_regex(p)
        d = next(d for d in deps if d["name"] == "rich")
        assert d["version"] is None



    def test_requirements_txt_found(self, tmp_path):
        _write(tmp_path, "requirements.txt", "requests\nnumpy\n")
        deps = _parse_python_deps(tmp_path)
        names = [d["name"] for d in deps]
        assert "requests" in names and "numpy" in names

    def test_requirements_dev_txt_found(self, tmp_path):
        _write(tmp_path, "requirements-dev.txt", "pytest\nblack\n")
        deps = _parse_python_deps(tmp_path)
        assert "pytest" in [d["name"] for d in deps]

    def test_nested_requirements_found(self, tmp_path):
        _write(tmp_path, "requirements/base.txt", "django\n")
        deps = _parse_python_deps(tmp_path)
        assert "django" in [d["name"] for d in deps]

    def test_no_manifest_returns_empty(self, tmp_path):
        assert _parse_python_deps(tmp_path) == []

    def test_setup_cfg_install_requires(self, tmp_path):
        content = "[options]\ninstall_requires =\n    requests>=2.0\n    numpy\n"
        _write(tmp_path, "setup.cfg", content)
        deps = _parse_python_deps(tmp_path)
        names = [d["name"] for d in deps]
        assert "requests" in names and "numpy" in names

    def test_pyproject_toml_found(self, tmp_path):
        _write(tmp_path, "pyproject.toml", '[project]\ndependencies = ["requests"]\n')
        deps = _parse_python_deps(tmp_path)
        assert "requests" in [d["name"] for d in deps]

    def test_pyproject_toml_combined_with_requirements_txt(self, tmp_path):
        _write(tmp_path, "requirements.txt", "numpy\n")
        _write(tmp_path, "pyproject.toml", '[project]\ndependencies = ["requests"]\n')
        deps = _parse_python_deps(tmp_path)
        names = [d["name"] for d in deps]
        assert "numpy" in names and "requests" in names


class TestParseJsDeps:
    def test_main_dependencies(self, tmp_path):
        pkg = {"dependencies": {"express": "^4.18.0", "axios": "^1.0.0"}}
        _write(tmp_path, "package.json", json.dumps(pkg))
        deps = _parse_js_deps(tmp_path)
        names = [d["name"] for d in deps]
        assert "express" in names and "axios" in names

    def test_dev_dependencies_kind(self, tmp_path):
        pkg = {"devDependencies": {"jest": "^29.0.0"}}
        _write(tmp_path, "package.json", json.dumps(pkg))
        deps = _parse_js_deps(tmp_path)
        assert all(d["kind"] == "dev" for d in deps)

    def test_peer_dependencies_kind(self, tmp_path):
        pkg = {"peerDependencies": {"react": ">=17.0.0"}}
        _write(tmp_path, "package.json", json.dumps(pkg))
        deps = _parse_js_deps(tmp_path)
        assert all(d["kind"] == "optional" for d in deps)

    def test_version_preserved(self, tmp_path):
        pkg = {"dependencies": {"lodash": "^4.17.21"}}
        _write(tmp_path, "package.json", json.dumps(pkg))
        deps = _parse_js_deps(tmp_path)
        d = next(d for d in deps if d["name"] == "lodash")
        assert d["version"] == "^4.17.21"

    def test_no_package_json_returns_empty(self, tmp_path):
        assert _parse_js_deps(tmp_path) == []

    def test_empty_package_json(self, tmp_path):
        _write(tmp_path, "package.json", "{}")
        assert _parse_js_deps(tmp_path) == []

    def test_mixed_kinds(self, tmp_path):
        pkg = {"dependencies": {"express": "^4.0.0"}, "devDependencies": {"jest": "^29.0.0"}}
        _write(tmp_path, "package.json", json.dumps(pkg))
        deps = _parse_js_deps(tmp_path)
        kinds = {d["name"]: d["kind"] for d in deps}
        assert kinds["express"] == "main" and kinds["jest"] == "dev"


class TestParseJavaDeps:
    POM = """<project>
  <dependencies>
    <dependency>
      <groupId>org.springframework</groupId>
      <artifactId>spring-core</artifactId>
      <version>5.3.20</version>
    </dependency>
    <dependency>
      <groupId>junit</groupId>
      <artifactId>junit</artifactId>
      <version>4.13.2</version>
      <scope>test</scope>
    </dependency>
  </dependencies>
</project>"""

    def test_main_dependency_parsed(self, tmp_path):
        _write(tmp_path, "pom.xml", self.POM)
        deps = _parse_java_deps(tmp_path)
        assert "org.springframework:spring-core" in [d["name"] for d in deps]

    def test_test_scope_is_dev(self, tmp_path):
        _write(tmp_path, "pom.xml", self.POM)
        deps = _parse_java_deps(tmp_path)
        junit = next(d for d in deps if "junit" in d["name"])
        assert junit["kind"] == "dev"

    def test_version_preserved(self, tmp_path):
        _write(tmp_path, "pom.xml", self.POM)
        deps = _parse_java_deps(tmp_path)
        spring = next(d for d in deps if "spring-core" in d["name"])
        assert spring["version"] == "5.3.20"

    def test_no_pom_returns_empty(self, tmp_path):
        assert _parse_java_deps(tmp_path) == []

    def test_pom_in_subdirectory(self, tmp_path):
        _write(tmp_path, "myapp/pom.xml", self.POM)
        assert len(_parse_java_deps(tmp_path)) >= 1


class TestParseCppDeps:
    def test_conanfile_txt_parsed(self, tmp_path):
        _write(tmp_path, "conanfile.txt", "[requires]\nboost/1.81.0\nopenssl/3.1.0\n")
        deps = _parse_cpp_deps(tmp_path)
        names = [d["name"] for d in deps]
        assert "boost" in names and "openssl" in names

    def test_conan_version_preserved(self, tmp_path):
        _write(tmp_path, "conanfile.txt", "[requires]\nboost/1.81.0\n")
        deps = _parse_cpp_deps(tmp_path)
        d = next(d for d in deps if d["name"] == "boost")
        assert d["version"] == "1.81.0"

    def test_vcpkg_json_parsed(self, tmp_path):
        content = json.dumps({"dependencies": ["boost", {"name": "openssl", "version-semver": "3.1.0"}]})
        _write(tmp_path, "vcpkg.json", content)
        deps = _parse_cpp_deps(tmp_path)
        names = [d["name"] for d in deps]
        assert "boost" in names and "openssl" in names

    def test_cmake_find_package_parsed(self, tmp_path):
        _write(tmp_path, "CMakeLists.txt", "find_package(OpenSSL REQUIRED)\nfind_package(Boost REQUIRED)\n")
        deps = _parse_cpp_deps(tmp_path)
        names = [d["name"] for d in deps]
        assert "OpenSSL" in names and "Boost" in names

    def test_cmake_ignores_system_packages(self, tmp_path):
        _write(tmp_path, "CMakeLists.txt", "find_package(CMake)\nfind_package(Threads)\n")
        deps = _parse_cpp_deps(tmp_path)
        names = [d["name"] for d in deps]
        assert "CMake" not in names and "Threads" not in names

    def test_no_manifest_returns_empty(self, tmp_path):
        assert _parse_cpp_deps(tmp_path) == []


class TestParseDependencies:
    def test_python_dispatches_correctly(self, tmp_path):
        _write(tmp_path, "requirements.txt", "requests\n")
        deps = parse_dependencies(str(tmp_path), "Python")
        assert any(d["name"] == "requests" for d in deps)

    def test_javascript_dispatches_correctly(self, tmp_path):
        _write(tmp_path, "package.json", json.dumps({"dependencies": {"express": "^4.0.0"}}))
        deps = parse_dependencies(str(tmp_path), "JavaScript")
        assert any(d["name"] == "express" for d in deps)

    def test_java_dispatches_correctly(self, tmp_path):
        pom = "<project><dependencies><dependency><groupId>org.test</groupId><artifactId>lib</artifactId><version>1.0</version></dependency></dependencies></project>"
        _write(tmp_path, "pom.xml", pom)
        deps = parse_dependencies(str(tmp_path), "Java")
        assert any("org.test" in d["name"] for d in deps)

    def test_cpp_dispatches_correctly(self, tmp_path):
        _write(tmp_path, "conanfile.txt", "[requires]\nboost/1.81.0\n")
        deps = parse_dependencies(str(tmp_path), "C++")
        assert any(d["name"] == "boost" for d in deps)

    def test_unknown_language_returns_empty(self, tmp_path):
        assert parse_dependencies(str(tmp_path), "COBOL") == []

    def test_returns_list_of_dicts(self, tmp_path):
        _write(tmp_path, "requirements.txt", "requests\nnumpy\n")
        deps = parse_dependencies(str(tmp_path), "Python")
        assert isinstance(deps, list)
        for d in deps:
            assert {"name", "version", "kind"} <= d.keys()

    def test_kind_values_are_valid(self, tmp_path):
        pkg = {
            "dependencies":     {"express": "^4.0.0"},
            "devDependencies":  {"jest":    "^29.0.0"},
            "peerDependencies": {"react":   ">=17.0.0"},
        }
        _write(tmp_path, "package.json", json.dumps(pkg))
        deps = parse_dependencies(str(tmp_path), "JavaScript")
        for d in deps:
            assert d["kind"] in {"main", "dev", "optional"}