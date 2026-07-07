"""Unit tests for the multi-agent orchestration, parsing in isolation.

No model or database is needed, so these run in CI without Ollama or Postgres.
"""

from src.agent import multiagent


def test_parse_extracts_json_object():
    parsed = multiagent._parse('noise {"agents": ["document", "web"]} tail', {})
    assert parsed == {"agents": ["document", "web"]}


def test_parse_falls_back_on_bad_json():
    default = {"agents": ["document"]}
    assert multiagent._parse("not json at all", default) == default
