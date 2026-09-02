"""
sieve/core/dependencies.py

Parses direct package dependencies from repository manifest files.

Supported manifest formats:
  Python:     requirements.txt, requirements/*.txt, pyproject.toml (PEP 508)
  JavaScript: package.json (dependencies + devDependencies)
  Java:       pom.xml (Maven <dependency> blocks)
  C++:        conanfile.txt, conanfile.py, vcpkg.json

Returns a list of dicts: [{name, version, kind}]
  name:    package name (str)
  version: version constraint string or None
  kind:    "main" | "dev" | "optional"
"""

import re
from pathlib import Path
from typing import Optional


# ─── Python ───────────────────────────────────────────────────────────────────

def _parse_requirements_txt(path: Path) -> list[dict]:
    """Parse a requirements.txt file."""
    deps = []
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or line.startswith("-"):
            continue
        # Strip inline comments
        line = line.split("#")[0].strip()
        if not line:
            continue
        # Strip extras specifier (e.g. requests[security] -> requests)
        line_no_extras = re.sub(r"\[[^\]]*\]", "", line).strip()
        # Parse name and version constraint
        m = re.match(r"^([A-Za-z0-9_\-\.]+)\s*([><=!~^,\s\d\.\*]+)?", line_no_extras)
        if m:
            name    = m.group(1).strip()
            version = m.group(2).strip() if m.group(2) else None
            deps.append({"name": name, "version": version or None, "kind": "main"})
    return deps


def _parse_pyproject_toml(path: Path) -> list[dict]:
    """Parse dependencies from pyproject.toml."""
    import tomllib
    deps = []
    try:
        data = tomllib.loads(path.read_text(encoding="utf-8", errors="replace"))
        project = data.get("project", {})

        for dep in project.get("dependencies", []):
            dep_clean = re.sub(r"\[[^\]]*\]", "", dep).strip()
            m = re.match(r"^([A-Za-z0-9_\-\.]+)\s*([><=!~^,\s\d\.\*a-zA-Z]+)?", dep_clean)
            if m:
                deps.append({
                    "name":    m.group(1).strip(),
                    "version": m.group(2).strip() if m.group(2) else None,
                    "kind":    "main",
                })

        opt = project.get("optional-dependencies", {})
        for group, group_deps in opt.items():
            for dep in group_deps:
                dep_clean = re.sub(r"\[[^\]]*\]", "", dep).strip()
                m = re.match(r"^([A-Za-z0-9_\-\.]+)\s*([><=!~^,\s\d\.\*a-zA-Z]+)?", dep_clean)
                if m:
                    deps.append({
                        "name":    m.group(1).strip(),
                        "version": m.group(2).strip() if m.group(2) else None,
                        "kind":    "optional",
                    })

        # Poetry format
        poetry = data.get("tool", {}).get("poetry", {})
        for dep, spec in poetry.get("dependencies", {}).items():
            if dep.lower() == "python":
                continue
            version = spec if isinstance(spec, str) else (
                spec.get("version") if isinstance(spec, dict) else None
            )
            deps.append({"name": dep, "version": version, "kind": "main"})
        for dep, spec in poetry.get("dev-dependencies", {}).items():
            version = spec if isinstance(spec, str) else (
                spec.get("version") if isinstance(spec, dict) else None
            )
            deps.append({"name": dep, "version": version, "kind": "dev"})

    except Exception:
        return _parse_pyproject_toml_regex(path)

    return deps


def _parse_pyproject_toml_regex(path: Path) -> list[dict]:
    """Regex fallback for pyproject.toml."""
    deps = []
    text = path.read_text(encoding="utf-8", errors="replace")
    m = re.search(r'dependencies\s*=\s*\[([^\]]*)\]', text, re.DOTALL)
    if m:
        for item in re.findall(r'"([^"]+)"', m.group(1)):
            item_clean = re.sub(r"\[[^\]]*\]", "", item).strip()
            pm = re.match(r"^([A-Za-z0-9_\-\.]+)\s*([><=!~^,\s\d\.\*]+)?", item_clean)
            if pm:
                deps.append({
                    "name":    pm.group(1).strip(),
                    "version": pm.group(2).strip() or None,
                    "kind":    "main",
                })
    return deps


def _parse_python_deps(repo_path: Path) -> list[dict]:
    deps = []

    # requirements.txt and requirements/*.txt
    for req_file in list(repo_path.glob("requirements*.txt")) + \
                    list(repo_path.glob("requirements/*.txt")):
        try:
            deps.extend(_parse_requirements_txt(req_file))
        except Exception:
            pass

    # pyproject.toml
    ppt = repo_path / "pyproject.toml"
    if ppt.exists():
        try:
            deps.extend(_parse_pyproject_toml(ppt))
        except Exception:
            pass

    # setup.cfg install_requires
    setup_cfg = repo_path / "setup.cfg"
    if setup_cfg.exists():
        try:
            text = setup_cfg.read_text(encoding="utf-8", errors="replace")
            m = re.search(r"install_requires\s*=\s*((?:\n\s+[^\n]+)+)", text)
            if m:
                for line in m.group(1).strip().splitlines():
                    line = re.sub(r"\[[^\]]*\]", "", line.strip()).strip()
                    pm = re.match(r"^([A-Za-z0-9_\-\.]+)\s*([><=!~^,\s\d\.\*]+)?", line)
                    if pm and pm.group(1):
                        deps.append({
                            "name":    pm.group(1).strip(),
                            "version": pm.group(2).strip() if pm.group(2) else None,
                            "kind":    "main",
                        })
        except Exception:
            pass

    return deps


# ─── JavaScript ───────────────────────────────────────────────────────────────

def _parse_js_deps(repo_path: Path) -> list[dict]:
    deps = []
    pkg = repo_path / "package.json"
    if not pkg.exists():
        return deps
    try:
        import json
        data = json.loads(pkg.read_text(encoding="utf-8", errors="replace"))
        for name, version in data.get("dependencies", {}).items():
            deps.append({"name": name, "version": version, "kind": "main"})
        for name, version in data.get("devDependencies", {}).items():
            deps.append({"name": name, "version": version, "kind": "dev"})
        for name, version in data.get("peerDependencies", {}).items():
            deps.append({"name": name, "version": version, "kind": "optional"})
    except Exception:
        pass
    return deps


# ─── Java ─────────────────────────────────────────────────────────────────────

def _parse_java_deps(repo_path: Path) -> list[dict]:
    deps = []
    pom = repo_path / "pom.xml"
    if not pom.exists():
        # Search one level deep
        poms = list(repo_path.glob("*/pom.xml"))
        if poms:
            pom = poms[0]
        else:
            return deps

    try:
        text = pom.read_text(encoding="utf-8", errors="replace")
        # Extract <dependency> blocks
        for block in re.finditer(
            r"<dependency>(.*?)</dependency>", text, re.DOTALL
        ):
            b     = block.group(1)
            gid   = re.search(r"<groupId>(.*?)</groupId>",       b)
            aid   = re.search(r"<artifactId>(.*?)</artifactId>", b)
            ver   = re.search(r"<version>(.*?)</version>",       b)
            scope = re.search(r"<scope>(.*?)</scope>",           b)

            if gid and aid:
                name    = f"{gid.group(1).strip()}:{aid.group(1).strip()}"
                version = ver.group(1).strip() if ver else None
                kind    = "dev" if scope and scope.group(1).strip().lower() in (
                    "test", "provided"
                ) else "main"
                deps.append({"name": name, "version": version, "kind": kind})
    except Exception:
        pass
    return deps


# ─── C++ ──────────────────────────────────────────────────────────────────────

def _parse_cpp_deps(repo_path: Path) -> list[dict]:
    deps = []

    # conanfile.txt
    conan_txt = repo_path / "conanfile.txt"
    if conan_txt.exists():
        try:
            text = conan_txt.read_text(encoding="utf-8", errors="replace")
            in_requires = False
            for line in text.splitlines():
                line = line.strip()
                if line.lower() == "[requires]":
                    in_requires = True
                    continue
                if line.startswith("["):
                    in_requires = False
                if in_requires and line and not line.startswith("#"):
                    m = re.match(r"^([A-Za-z0-9_\-\.]+)/([^\s]+)", line)
                    if m:
                        deps.append({
                            "name":    m.group(1),
                            "version": m.group(2),
                            "kind":    "main",
                        })
        except Exception:
            pass

    # vcpkg.json
    vcpkg = repo_path / "vcpkg.json"
    if vcpkg.exists():
        try:
            import json
            data = json.loads(vcpkg.read_text(encoding="utf-8", errors="replace"))
            for dep in data.get("dependencies", []):
                if isinstance(dep, str):
                    deps.append({"name": dep, "version": None, "kind": "main"})
                elif isinstance(dep, dict):
                    deps.append({
                        "name":    dep.get("name", ""),
                        "version": dep.get("version-semver") or dep.get("version"),
                        "kind":    "main",
                    })
        except Exception:
            pass

    # CMakeLists.txt — find_package() calls
    cmake = repo_path / "CMakeLists.txt"
    if cmake.exists():
        try:
            text = cmake.read_text(encoding="utf-8", errors="replace")
            for m in re.finditer(r"find_package\s*\(\s*(\w+)", text, re.IGNORECASE):
                name = m.group(1)
                if name.upper() not in {"CMAKE", "THREADS", "PKGCONFIG"}:
                    deps.append({"name": name, "version": None, "kind": "main"})
        except Exception:
            pass

    return deps


# ─── Public API ───────────────────────────────────────────────────────────────

def parse_dependencies(repo_path: str, language: str) -> list[dict]:
    """
    Parse direct package dependencies from a cloned repository.

    Args:
        repo_path: Absolute path to the cloned repository
        language:  One of Python, Java, JavaScript, C++

    Returns:
        List of dicts with keys: name (str), version (str|None), kind (str)
        kind is one of: "main", "dev", "optional"
        Returns empty list if no manifest found or parsing fails.
    """
    root = Path(repo_path)
    try:
        if language == "Python":
            return _parse_python_deps(root)
        elif language == "JavaScript":
            return _parse_js_deps(root)
        elif language == "Java":
            return _parse_java_deps(root)
        elif language == "C++":
            return _parse_cpp_deps(root)
    except Exception:
        pass
    return []