"""Chat interface for the RagFlowProMax Streamlit app.

Shows the answer plus an expandable trace of which agents ran (supervisor,
document, web, synthesizer, verifier), so the multi-agent coordination is
visible rather than hidden.
"""

import streamlit as st

from frontend import api_utils

_AGENT_LABEL = {
    "supervisor": "Supervisor, plan and route",
    "document": "Document agent, self correcting RAG",
    "web": "Web agent",
    "synthesizer": "Synthesizer",
    "verifier": "Verifier",
}


def _render_trace(steps, agents, verified) -> None:
    if not steps:
        return
    path = ", ".join(
        _AGENT_LABEL.get(s.get("agent"), s.get("agent", "")) for s in steps
    )
    header = f"Agents: {len(steps)} steps"
    if agents:
        header += f", routed to {' and '.join(agents)}"
    header += ", verified" if verified else ", not verified"
    with st.expander(header, expanded=False):
        st.caption(path)
        for index, step in enumerate(steps, 1):
            name = _AGENT_LABEL.get(step.get("agent"), step.get("agent", ""))
            detail = {key: value for key, value in step.items() if key != "agent"}
            st.markdown(f"**{index}. {name}**")
            if detail:
                st.json(detail, expanded=False)


def display_chat_interface() -> None:
    if "messages" not in st.session_state:
        st.session_state.messages = []
    if "session_id" not in st.session_state:
        st.session_state.session_id = None

    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
            if message.get("steps"):
                _render_trace(
                    message["steps"],
                    message.get("agents"),
                    message.get("verified", True),
                )

    prompt = st.chat_input("Ask a question about your documents")
    if not prompt:
        return

    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        model = st.session_state.get("model", "gpt-4o-mini")
        with st.spinner("Agents are working..."):
            try:
                result = api_utils.chat(prompt, st.session_state.session_id, model)
            except Exception as exc:
                st.error(f"Request failed: {exc}")
                return
        st.session_state.session_id = result.get("session_id")
        answer = result.get("answer", "")
        st.markdown(answer)
        _render_trace(
            result.get("steps", []),
            result.get("agents"),
            result.get("verified", True),
        )

    st.session_state.messages.append(
        {
            "role": "assistant",
            "content": answer,
            "steps": result.get("steps", []),
            "agents": result.get("agents"),
            "verified": result.get("verified", True),
        }
    )
