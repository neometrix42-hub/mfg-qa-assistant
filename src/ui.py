"""Streamlit UI. One file. Do not build a React frontend.

Week 7. Its only job is the screenshot/GIF in your README.
"""

import streamlit as st

from src.agent import ask

st.set_page_config(page_title="Manufacturing QA Assistant", page_icon="*")

st.title("Manufacturing QA Assistant")
st.caption("Ask about inspection data or quality procedures.")

with st.sidebar:
    st.subheader("Try")
    st.code("How often must the CMM be calibrated?", language=None)
    st.code("How many parts failed flatness in March 2026?", language=None)
    st.code("P-4417 failed flatness. What does the SOP require?", language=None)

question = st.text_input("Question", placeholder="Ask about inspection data or procedures...")

if question:
    with st.spinner("Thinking..."):
        result = ask(question)

    st.markdown(result["answer"])
    cols = st.columns(2)
    cols[0].metric("Tools used", ", ".join(result["tools_called"]) or "none")
    cols[1].metric("Latency", f"{result['latency_ms']} ms")

# Run: streamlit run src/ui.py
