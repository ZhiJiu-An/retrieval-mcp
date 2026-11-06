"""版本号只应有一个事实来源。"""

from __future__ import annotations

import tomllib
from pathlib import Path

from rag_mcp import __version__


def test_package_version_matches_pyproject():
    pyproject = Path(__file__).resolve().parents[1] / "pyproject.toml"
    data = tomllib.loads(pyproject.read_text(encoding="utf-8"))

    assert __version__ == data["project"]["version"]
