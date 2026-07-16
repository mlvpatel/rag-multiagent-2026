"""Multi-agent orchestration for rag-multiagent-2026 (supervisor pattern).

A supervisor routes the question to specialist worker agents, a synthesizer
merges their findings, and a verifier checks the answer is grounded. Every
agent's contribution is recorded in a trace, which is the observability spine
from the enterprise design. The whole thing is bounded, so it always terminates.

Keyless on Ollama. The document worker reuses the rag-agentic-2025 self correcting
RAG (so it is an agent of agents), and the web worker uses the grounded web tool.
"""

import json
from typing import Any, Dict, List, Tuple

from langchain_core.messages import HumanMessage, SystemMessage

from src.agent.graph import run_agent as run_document_agent
from src.agent.tools import web_search_docs
from src.core.config import settings
from src.core.langchain_utils import (
    _format_context,
    _make_llm,
    _reformulate_query,
    _to_lc_messages,
)

_SUPERVISOR_SYSTEM = (
    "You are a supervisor routing a question to specialist agents. Available "
    "agents: document (searches the internal documents) and web (searches the "
    'public web). Return ONLY compact JSON: {"agents": ["document"] or '
    '["document", "web"], "reason": "short"}. Use web only when the question '
    "likely needs current or external information the documents would not cover."
)

_SYNTH_SYSTEM = (
    "You are a synthesizer. Combine the specialist findings into one clear, "
    "grounded answer, using only the findings. If they do not contain the "
    "answer, say you do not have that information rather than inventing one."
)

_VERIFY_SYSTEM = (
    "You verify that an answer is grounded in the findings. Return ONLY compact "
    'JSON: {"verified": true or false, "reason": "short"}.'
)


def _parse(text: str, default: Dict[str, Any]) -> Dict[str, Any]:
    try:
        return json.loads(text[text.index("{") : text.rindex("}") + 1])
    except Exception:
        return default


def _verify(llm, answer: str, findings_text: str) -> Tuple[bool, str]:
    """Ask the verifier whether the answer is grounded in the findings.

    Fails CLOSED: a verdict that cannot be parsed counts as not verified. The
    verifier exists to catch bad output, and an unreadable verdict is bad
    output, not a pass.
    """
    vraw = llm.invoke(
        [
            SystemMessage(content=_VERIFY_SYSTEM),
            HumanMessage(content=f"Answer: {answer}\n\nFindings:\n{findings_text}"),
        ]
    ).content
    data = _parse(vraw, {"verified": False, "reason": "unparseable verdict"})
    return bool(data.get("verified", False)), data.get("reason", "")


def run_multiagent(model: str, question: str, chat_history=None) -> dict:
    steps: List[Dict[str, Any]] = []
    llm = _make_llm(model, temperature=0)

    # 0. Resolve the follow-up once, so the supervisor, every worker, the
    # synthesizer, and the verifier all see the same standalone question.
    history = _to_lc_messages(chat_history)
    query = _reformulate_query(llm, question, history)
    if query != question:
        steps.append({"agent": "reformulate", "standalone_question": query})

    # 1. Supervisor plans which specialists to use.
    raw = llm.invoke(
        [SystemMessage(content=_SUPERVISOR_SYSTEM), HumanMessage(content=query)]
    ).content
    plan = _parse(raw, {"agents": ["document"]})
    agents = plan.get("agents") or ["document"]
    if "document" not in agents:
        agents = ["document"] + agents
    steps.append(
        {"agent": "supervisor", "plan": agents, "reason": plan.get("reason", "")}
    )

    findings: Dict[str, Dict[str, Any]] = {}

    # 2. Document worker: the rag-agentic-2025 self correcting RAG over pgvector.
    # The question is already standalone, so no history is passed down.
    doc = run_document_agent(model, query, None)
    findings["document"] = {
        "answer": doc.get("answer", ""),
        "sources": sorted(
            {d.metadata.get("filename") for d in doc.get("context", []) if d.metadata}
        ),
        "confidence": doc.get("confidence"),
    }
    steps.append({"agent": "document", "confidence": doc.get("confidence")})

    # 3. Web worker, only when planned and explicitly enabled (grounded first).
    if "web" in agents and settings.agent_enable_web:
        web_docs = web_search_docs(query)
        context = _format_context(web_docs)
        if context:
            web_answer = llm.invoke(
                [
                    SystemMessage(
                        content="Answer only from these web snippets.\n\n" + context
                    ),
                    HumanMessage(content=query),
                ]
            ).content
        else:
            web_answer = "No web results were available."
        findings["web"] = {
            "answer": web_answer,
            "sources": [d.metadata.get("filename", "web") for d in web_docs],
        }
        steps.append({"agent": "web", "results": len(web_docs)})

    # 4. Synthesize the specialist findings into one answer.
    findings_text = "\n\n".join(
        f"[{name}] {f['answer']}" for name, f in findings.items()
    )
    answer = llm.invoke(
        [
            SystemMessage(content=_SYNTH_SYSTEM),
            HumanMessage(content=f"Question: {query}\n\nFindings:\n{findings_text}"),
        ]
    ).content
    steps.append({"agent": "synthesizer", "answer_chars": len(answer)})

    # 5. Verify, and GATE on the verdict: an unverified answer is synthesized
    # once more with the failure fed back, then re-verified. The final flag is
    # returned either way, so a caller can still see a second failure.
    verified, reason = _verify(llm, answer, findings_text)
    steps.append({"agent": "verifier", "verified": verified, "reason": reason})
    if not verified:
        answer = llm.invoke(
            [
                SystemMessage(content=_SYNTH_SYSTEM),
                HumanMessage(
                    content=(
                        f"Question: {query}\n\nFindings:\n{findings_text}\n\n"
                        f"Your previous answer was rejected by a grounding check "
                        f"({reason or 'not grounded in the findings'}). Rewrite the "
                        f"answer using ONLY the findings, and say the information "
                        f"is not available if the findings do not contain it."
                    )
                ),
            ]
        ).content
        steps.append(
            {"agent": "synthesizer", "retry": True, "answer_chars": len(answer)}
        )
        verified, reason = _verify(llm, answer, findings_text)
        steps.append({"agent": "verifier", "verified": verified, "reason": reason})

    sources = sorted(
        {s for f in findings.values() for s in (f.get("sources") or []) if s}
    )
    return {
        "answer": answer,
        "sources": sources,
        "agents": agents,
        "steps": steps,
        "verified": verified,
    }
