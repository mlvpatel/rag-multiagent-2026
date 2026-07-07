"""Tools available to the agent.

The web fallback is grounded first and off by default. It is a pluggable
extension point, not a hard dependency: when disabled or when the backend is
unavailable it returns nothing, so the agent stays strictly grounded in the
indexed documents unless web search is explicitly turned on.
"""

import logging
from typing import List

from langchain_core.documents import Document

from src.core.config import settings

logger = logging.getLogger(__name__)


def web_search_docs(query: str, k: int = 4) -> List[Document]:
    """Return web result snippets as Documents, or an empty list.

    Keyless via DuckDuckGo. Runs only when AGENT_ENABLE_WEB is true.
    """
    if not settings.agent_enable_web:
        return []
    try:
        from ddgs import DDGS

        results = DDGS().text(query, max_results=k)
        return [
            Document(
                page_content=r.get("body", ""),
                metadata={"filename": r.get("href", "web"), "source": "web"},
            )
            for r in results
        ]
    except Exception as exc:
        logger.warning("Web search unavailable: %s", exc)
        return []
