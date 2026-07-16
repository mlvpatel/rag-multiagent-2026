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


class _ScriptedLLM:
    """Plain fake chat model returning scripted replies in order."""

    def __init__(self, replies):
        self.replies = list(replies)
        self.prompts = []

    def invoke(self, messages, **kwargs):
        from langchain_core.messages import AIMessage

        self.prompts.append(messages[-1].content)
        return AIMessage(content=self.replies.pop(0))


def _doc_finding(*_a, **_k):
    return {"answer": "doc answer", "context": [], "confidence": 0.9}


def test_verifier_gates_and_regenerates_once(monkeypatch):
    """An unverified answer must be synthesized again with the failure fed
    back, then re-verified; the final flag reflects the second verdict."""
    import src.agent.multiagent as ma

    llm = _ScriptedLLM(
        [
            '{"agents": ["document"], "reason": "internal"}',  # supervisor
            "first answer",  # synth
            '{"verified": false, "reason": "unsupported claim"}',  # verify 1
            "second answer",  # synth retry
            '{"verified": true, "reason": "ok"}',  # verify 2
        ]
    )
    monkeypatch.setattr(ma, "_make_llm", lambda *a, **k: llm)
    monkeypatch.setattr(ma, "run_document_agent", _doc_finding)

    result = ma.run_multiagent("m", "what is the vacation policy?")

    assert result["answer"] == "second answer"
    assert result["verified"] is True
    verdicts = [s for s in result["steps"] if s["agent"] == "verifier"]
    assert [v["verified"] for v in verdicts] == [False, True]
    assert "rejected by a grounding check" in llm.prompts[3]


def test_unparseable_verdict_fails_closed(monkeypatch):
    """A verifier reply that is not JSON is a failure, not a pass."""
    import src.agent.multiagent as ma

    llm = _ScriptedLLM(
        [
            '{"agents": ["document"]}',
            "first answer",
            "sorry, I cannot produce JSON today",  # unparseable verdict 1
            "second answer",
            "still not json",  # unparseable verdict 2
        ]
    )
    monkeypatch.setattr(ma, "_make_llm", lambda *a, **k: llm)
    monkeypatch.setattr(ma, "run_document_agent", _doc_finding)

    result = ma.run_multiagent("m", "question")

    assert result["verified"] is False
    assert result["answer"] == "second answer"


def test_verified_answer_returns_without_retry(monkeypatch):
    import src.agent.multiagent as ma

    llm = _ScriptedLLM(
        [
            '{"agents": ["document"]}',
            "the answer",
            '{"verified": true, "reason": "grounded"}',
        ]
    )
    monkeypatch.setattr(ma, "_make_llm", lambda *a, **k: llm)
    monkeypatch.setattr(ma, "run_document_agent", _doc_finding)

    result = ma.run_multiagent("m", "question")

    assert result["verified"] is True
    assert result["answer"] == "the answer"
    assert len([s for s in result["steps"] if s["agent"] == "synthesizer"]) == 1


def test_followup_is_reformulated_for_all_agents(monkeypatch):
    """The supervisor and the document worker must see the standalone
    question, and the worker gets no history (already resolved)."""
    import src.agent.multiagent as ma

    llm = _ScriptedLLM(
        [
            '{"agents": ["document"]}',  # supervisor
            "answer",  # synth
            '{"verified": true}',  # verify
        ]
    )
    seen = {}

    def doc_agent(model, q, history):
        seen["query"], seen["history"] = q, history
        return {"answer": "doc", "context": [], "confidence": 0.8}

    monkeypatch.setattr(ma, "_make_llm", lambda *a, **k: llm)
    monkeypatch.setattr(ma, "run_document_agent", doc_agent)
    # the rewrite itself is unit tested in langchain_utils; here only the
    # wiring matters, so stub it to a known standalone question
    monkeypatch.setattr(
        ma,
        "_reformulate_query",
        lambda llm, q, h: "what is the price of Nimbus Pro?" if h else q,
    )

    history = [{"role": "human", "content": "tell me about Nimbus Pro"}]
    ma.run_multiagent("m", "and its price?", history)

    assert seen["query"] == "what is the price of Nimbus Pro?"
    assert seen["history"] is None
    assert llm.prompts[0] == "what is the price of Nimbus Pro?"
