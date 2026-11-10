"""
tests/test_version_consistency.py

The package version is stored in three authoritative places (pyproject.toml,
CITATION.cff, src/sieve/__init__.py). They must agree, be valid PEP 440, and the
two user-visible consumers (the export manifest and the Documentation page
footer) must take the value from sieve.__version__ rather than a literal.
"""

import json
import re
import tomllib
from pathlib import Path

from packaging.version import Version
from streamlit.testing.v1 import AppTest

import sieve
from sieve.core.export import export_dataset

ROOT = Path(__file__).resolve().parents[1]
DOC_PAGE = str(ROOT / "src" / "sieve" / "ui" / "pages" / "1_Documentation.py")


def _pyproject_version() -> str:
    with (ROOT / "pyproject.toml").open("rb") as f:
        return tomllib.load(f)["project"]["version"]


def _citation_version() -> str:
    """Strict line-based read of the top-level `version:` key (PyYAML is not a dependency)."""
    lines = (ROOT / "CITATION.cff").read_text(encoding="utf-8").splitlines()
    # Top-level only: no leading whitespace, so preferred-citation's nested keys cannot match.
    values = [m.group(1) for line in lines if (m := re.fullmatch(r"version:\s*(\S.*?)\s*", line))]
    assert len(values) == 1, f"expected exactly one top-level 'version:' in CITATION.cff, found {values}"
    value = values[0]
    if len(value) >= 2 and value[0] == value[-1] and value[0] in "\"'":
        value = value[1:-1]
    return value


def test_three_sources_agree():
    versions = {
        "pyproject.toml": _pyproject_version(),
        "CITATION.cff": _citation_version(),
        "sieve.__version__": sieve.__version__,
    }
    assert len(set(versions.values())) == 1, versions


def test_version_is_valid_pep440():
    # Version() raises InvalidVersion on anything that is not PEP 440.
    parsed = Version(sieve.__version__)
    assert str(parsed) == sieve.__version__, "version is valid but not in normalized PEP 440 form"


def test_manifest_sieve_version_comes_from_the_package(tmp_path):
    paths = export_dataset([], [], str(tmp_path), "jsonl", {"language": "Python"}, [])
    manifest = json.loads(Path(paths["manifest"]).read_text())
    assert manifest["sieve_version"] == sieve.__version__


def test_documentation_footer_comes_from_the_package():
    at = AppTest.from_file(DOC_PAGE, default_timeout=60).run()
    assert not at.exception, [e.value for e in at.exception]
    footers = [c.value for c in at.caption if c.value.startswith("SIEVE v")]
    assert len(footers) == 1, footers
    assert footers[0].startswith(f"SIEVE v{sieve.__version__} ")
