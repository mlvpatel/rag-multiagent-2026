"""Graph state for the rag-agentic-2025 agentic RAG.

The state is the execution-tracking board from the planning diagram: it carries
the question, the current search query, the retrieved documents, the confidence
grade, bounded attempt counters, and a running list of steps that doubles as the
agent's trace for observability and the UI. The steps list uses an add reducer
so every node appends to it rather than overwriting.
"""

from __future__ import annotations

import operator
from typing import Annotated, Any, Dict, List

from langchain_core.documents import Document
from typing_extensions import TypedDict


class AgentState(TypedDict, total=False):
    model: str
    question: str
    chat_history: List[Any]
    query: str
    documents: List[Document]
    confidence: float
    grade: str
    attempts: int
    generations: int
    used_web: bool
    answer: str
    grounded: bool
    steps: Annotated[List[Dict[str, Any]], operator.add]
