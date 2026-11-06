"""统一的环境变量解析：非法/越界值必须回退默认值。"""

from __future__ import annotations

from rag_mcp.env import env_bool, env_float, env_int


def test_env_int_uses_minimum(monkeypatch):
    monkeypatch.setenv("TEST_INT", "0")
    assert env_int("TEST_INT", 7, minimum=1) == 7
    monkeypatch.setenv("TEST_INT", "3")
    assert env_int("TEST_INT", 7, minimum=1) == 3


def test_env_float_range(monkeypatch):
    monkeypatch.setenv("TEST_FLOAT", "1.5")
    assert env_float("TEST_FLOAT", 0.5, maximum=1.0) == 0.5
    monkeypatch.setenv("TEST_FLOAT", "0.75")
    assert env_float("TEST_FLOAT", 0.5, minimum=0.0, maximum=1.0) == 0.75


def test_env_bool_accepts_common_true_values(monkeypatch):
    for value in ("1", "true", "yes", "on"):
        monkeypatch.setenv("TEST_BOOL", value)
        assert env_bool("TEST_BOOL", False) is True
    monkeypatch.setenv("TEST_BOOL", "off")
    assert env_bool("TEST_BOOL", True) is False
