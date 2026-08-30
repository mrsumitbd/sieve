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


class TestParsePythonDeps:
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