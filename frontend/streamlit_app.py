"""RagFlowPro Streamlit application entry point."""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import streamlit as st  # noqa: E402

from frontend.chat_interface import display_chat_interface  # noqa: E402
from frontend.sidebar import display_sidebar  # noqa: E402


def main() -> None:
    st.set_page_config(page_title="RagFlowProMax", layout="wide")
    st.title("RagFlowProMax")
    st.caption(
        "Multi agent enterprise RAG, 2026. A supervisor routes specialist agents and a verifier checks the answer."
    )
    display_sidebar()
    display_chat_interface()


if __name__ == "__main__":
    main()
