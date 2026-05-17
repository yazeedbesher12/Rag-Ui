from __future__ import annotations

import streamlit as st

from app.rag import get_settings, init_rag, run_query


st.set_page_config(
    page_title="CV RAG Assistant",
    page_icon=":material/description:",
    layout="centered",
)


@st.cache_resource(show_spinner=False)
def load_rag() -> None:
    init_rag()


def render_context_items(context: list[dict]) -> None:
    for index, item in enumerate(context, start=1):
        metadata = item.get("metadata") or {}
        source = (
            metadata.get("source")
            or metadata.get("file_name")
            or metadata.get("name")
            or metadata.get("id")
            or "Unknown source"
        )
        score = metadata.get("score")
        score_label = f" | score: {score:.4f}" if isinstance(score, float) else ""

        with st.expander(f"Chunk {index}: {source}{score_label}"):
            st.write(item.get("content", ""))
            if metadata:
                st.caption("Metadata")
                st.json(metadata)


st.title("CV RAG Assistant")
st.caption("Ask questions about the CVs indexed in Pinecone.")

with st.sidebar:
    st.header("Settings")
    include_context = st.toggle("Show retrieved context", value=False)

    try:
        settings = get_settings()
        st.success("Environment ready")
        st.caption(f"Index: {settings.pinecone_index}")
        st.caption(f"Model: {settings.openrouter_model}")
        st.caption(f"Top K: {settings.retriever_top_k}")
    except Exception as exc:
        st.error(str(exc))

question = st.text_area(
    "Question",
    placeholder="Example: Which candidate has Python and FastAPI experience?",
    height=110,
)

submitted = st.button("Ask", type="primary", use_container_width=True)

if submitted:
    clean_question = question.strip()
    if not clean_question:
        st.warning("Please enter a question first.")
    else:
        try:
            with st.spinner("Searching CVs and generating an answer..."):
                load_rag()
                result = run_query(clean_question, include_context=include_context)

            st.subheader("Answer")
            st.write(result["answer"])

            if include_context and result.get("context"):
                st.subheader("Retrieved Context")
                render_context_items(result["context"])
        except Exception as exc:
            st.error(f"Could not answer the question: {exc}")
