"""
Confidential Document Q&A Workbench — Streamlit entrypoint.

Document storage, chunking, embeddings, and retrieval remain local to this
machine. LLM inference (routing, summarization, Q&A generation, comparison)
is provided by the Google Gemini API and requires outbound network access
and a configured GEMINI_API_KEY. This application is NOT fully offline.
"""
import time

import streamlit as st

from ingestion.indexer import index_file
from models.embeddings import embedding_model_available
from models.gemini_client import health_check as gemini_health_check
from retrieval.vector_store import VectorStore, health_check as chroma_health_check
from utils.config import DEFAULT_EMBEDDING_MODEL, SUPPORTED_EXTENSIONS, get_settings
from utils.logger import get_logger
from graph.workflow import run_workflow

logger = get_logger(__name__)

st.set_page_config(page_title="Confidential Document Workbench", page_icon="🔒", layout="wide")

settings = get_settings()

if "agent_activity" not in st.session_state:
    st.session_state.agent_activity = []


@st.cache_resource(show_spinner=False)
def _get_vector_store(embedding_model: str):
    return VectorStore(embedding_model_name=embedding_model)


def vector_store() -> VectorStore:
    return _get_vector_store(settings.embedding_model)


def log_activity(entry: dict):
    st.session_state.agent_activity.insert(0, entry)
    st.session_state.agent_activity = st.session_state.agent_activity[:25]


# ---------------------------------------------------------------------------
# Sidebar navigation + local-only banner
# ---------------------------------------------------------------------------
st.sidebar.title("🔒 Confidential Workbench")
st.sidebar.caption("Documents and AI inference remain local to this machine.")
page = st.sidebar.radio(
    "Navigate",
    ["Dashboard", "Documents", "Ask Documents", "Summarize", "Compare", "Agent Activity", "Settings"],
)

st.sidebar.divider()
st.sidebar.markdown(
    "**Privacy notice**\n\n"
    "Documents, chunking, embeddings, and ChromaDB storage stay on this machine. "
    "LLM requests (routing, summarization, Q&A, comparison) are sent to the "
    "**Google Gemini API** over the network. This application is **not fully "
    "offline** and is not a certified/regulatory-compliant system — verify "
    "against your own organization's requirements before using it with real "
    "confidential data."
)


# ---------------------------------------------------------------------------
# Dashboard
# ---------------------------------------------------------------------------
def render_dashboard():
    st.title("Dashboard")
    st.caption(
        "Documents, embeddings, and ChromaDB remain local. LLM requests are sent to the Google Gemini API."
    )

    vs = vector_store()
    docs = vs.list_documents()
    chunk_count = vs.count_chunks()

    from models.gemini_client import api_key_configured
    key_configured = api_key_configured()
    chroma_ok, chroma_msg = chroma_health_check()
    embed_ok, embed_msg = embedding_model_available(settings.embedding_model)

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Documents indexed", len(docs))
    col2.metric("Total chunks", chunk_count)
    col3.metric("ChromaDB", "Online" if chroma_ok else "Offline")
    col4.metric("Gemini API Key", "Configured" if key_configured else "Missing")

    st.subheader("System Health")
    h1, h2 = st.columns(2)
    with h1:
        if key_configured:
            st.success("GEMINI_API_KEY is configured.")
        else:
            st.error(
                "GEMINI_API_KEY is not set. Create a `.env` file (see `.env.example`) with "
                "`GEMINI_API_KEY=<your key>`, or export it as an environment variable, then restart the app."
            )
        st.caption(f"Configured model: `{settings.gemini_model}`")
        if st.button("Test Gemini connection"):
            with st.spinner("Contacting Gemini API..."):
                status = gemini_health_check(model=settings.gemini_model)
            if status["gemini_reachable"]:
                st.success(f"Successfully reached Gemini model '{status['model']}'.")
            else:
                st.error(status["error"] or "Could not reach the Gemini API.")
    with h2:
        if chroma_ok:
            st.success(chroma_msg)
        else:
            st.error(chroma_msg)
        if embed_ok:
            st.success(embed_msg)
        else:
            st.error(embed_msg)

    st.subheader("Current Configuration")
    st.json({
        "LLM (Google Gemini)": settings.gemini_model,
        "Embedding model (local)": settings.embedding_model,
        "Chunk size": settings.chunk_size,
        "Chunk overlap": settings.chunk_overlap,
        "Top-K retrieval": settings.top_k,
        "Temperature": settings.temperature,
    })


# ---------------------------------------------------------------------------
# Documents
# ---------------------------------------------------------------------------
def render_documents():
    st.title("Documents")
    st.caption("Upload confidential PDF/DOCX files. Everything is processed and stored locally.")

    uploaded_files = st.file_uploader(
        "Upload PDF or DOCX documents",
        type=["pdf", "docx"],
        accept_multiple_files=True,
    )

    if uploaded_files:
        vs = vector_store()
        if st.button("Index uploaded files", type="primary"):
            progress = st.progress(0.0, text="Starting ingestion...")
            results = []
            for i, uf in enumerate(uploaded_files):
                progress.progress((i) / len(uploaded_files), text=f"Processing {uf.name}...")
                try:
                    file_bytes = uf.read()
                    result = index_file(
                        file_bytes=file_bytes,
                        filename=uf.name,
                        vector_store=vs,
                        chunk_size=settings.chunk_size,
                        chunk_overlap=settings.chunk_overlap,
                    )
                    results.append(result)
                except Exception as exc:  # noqa: BLE001
                    st.error(f"Unexpected error indexing '{uf.name}': {exc}")
            progress.progress(1.0, text="Done.")
            for r in results:
                if r.status == "indexed":
                    st.success(r.message)
                elif r.status == "duplicate":
                    st.info(r.message)
                else:
                    st.error(r.message)

    st.divider()
    st.subheader("Indexed Documents")

    vs = vector_store()
    docs = vs.list_documents()

    if not docs:
        st.info("No documents indexed yet. Upload a PDF or DOCX above.")
        return

    for doc in docs:
        with st.container(border=True):
            c1, c2, c3, c4 = st.columns([3, 2, 1, 1])
            c1.markdown(f"**{doc['filename']}**")
            c2.caption(f"SHA-256: {doc['sha256'][:16]}...")
            c3.caption(f"{doc['chunk_count']} chunks")
            with c4:
                if st.button("Delete", key=f"del_{doc['sha256']}"):
                    removed = vs.delete_document(doc["sha256"])
                    st.success(f"Deleted {removed} chunks for '{doc['filename']}'.")
                    st.rerun()
            if st.button("Re-index", key=f"reindex_{doc['sha256']}"):
                st.info("Re-index requires re-uploading the original file (raw bytes aren't retained in ChromaDB metadata).")


# ---------------------------------------------------------------------------
# Ask Documents
# ---------------------------------------------------------------------------
def render_ask():
    st.title("Ask Documents")
    st.caption("Automatic agent routing via the LangGraph Orchestrator (LLM-based intent classification).")

    vs = vector_store()
    docs = vs.list_documents()
    if not docs:
        st.warning("No documents indexed yet. Go to the Documents page first.")
        return

    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []

    # Document selection for summary/comparison requests from chat.
    # Q&A still searches all documents when nothing is selected.
    doc_options = {
        f"{d['filename']} ({d['sha256'][:8]})": d["sha256"]
        for d in docs
    }

    selected_labels = st.multiselect(
        "Select document(s) for summary or comparison",
        options=list(doc_options.keys()),
        help=(
            "Select one document when asking for a summary. "
            "Select two documents when asking to compare them. "
            "Leave empty for normal Q&A across all indexed documents."
        ),
    )

    selected_documents = [doc_options[label] for label in selected_labels]

    for turn in st.session_state.chat_history:
        with st.chat_message(turn["role"]):
            st.markdown(turn["content"])
            if turn.get("meta"):
                _render_explainability(turn["meta"])

    user_query = st.chat_input(
        "Ask a question, request a summary, or ask to compare documents..."
    )

    if user_query:
        st.session_state.chat_history.append(
            {"role": "user", "content": user_query}
        )

        with st.chat_message("user"):
            st.markdown(user_query)

        with st.chat_message("assistant"):
            with st.spinner("Routing and processing..."):
                try:
                    final_state = run_workflow(
                        user_query=user_query,
                        model=settings.gemini_model,
                        temperature=settings.temperature,
                        top_k=settings.top_k,
                        selected_documents=selected_documents,
                    )
                except Exception as exc:  # noqa: BLE001
                    st.error(f"The workflow encountered an unexpected error: {exc}")
                    logger.error(f"Workflow crash: {exc}")
                    final_state = None

            if final_state:
                st.markdown(final_state.get("final_response", ""))

                meta = {
                    "selected_agent": final_state.get("selected_agent"),
                    "routing_reason": final_state.get("routing_reason"),
                    "sources": final_state.get("sources", []),
                    "retrieved_context": final_state.get("retrieved_context", []),
                    "llm_model": final_state.get("llm_model"),
                    "processing_time_seconds": final_state.get("processing_time_seconds"),
                    "errors": final_state.get("errors", []),
                    "execution_trace": final_state.get("execution_trace", []),
                }

                _render_explainability(meta)

                st.session_state.chat_history.append(
                    {
                        "role": "assistant",
                        "content": final_state.get("final_response", ""),
                        "meta": meta,
                    }
                )

                log_activity({
                    "query": user_query,
                    "agent": meta["selected_agent"],
                    "reason": meta["routing_reason"],
                    "time": meta["processing_time_seconds"],
                    "errors": meta["errors"],
                })


def _render_explainability(meta: dict):
    with st.expander(f"🧭 Selected Agent: **{meta.get('selected_agent', 'n/a')}** — details"):
        st.markdown(f"**Routing reason:** {meta.get('routing_reason', 'n/a')}")
        st.markdown(f"**LLM model:** {meta.get('llm_model', 'n/a')}")
        st.markdown(f"**Processing time:** {meta.get('processing_time_seconds', 'n/a')}s")
        if meta.get("execution_trace"):
            st.markdown("**Execution trace:**")
            for step in meta["execution_trace"]:
                st.markdown(f"- {step}")
        if meta.get("sources"):
            st.markdown("**Sources:**")
            for s in meta["sources"]:
                page = f" — Page {s['page_number']}" if s.get("page_number") else ""
                st.markdown(f"- {s.get('filename')}{page}")
        if meta.get("retrieved_context"):
            st.markdown("**Retrieved context:**")
            for i, chunk in enumerate(meta["retrieved_context"]):
                m = chunk.get("metadata", {})
                page = f" — Page {m.get('page_number')}" if m.get("page_number") else ""
                with st.expander(f"Chunk {i+1}: {m.get('filename')}{page}"):
                    st.text(chunk.get("text", ""))
        if meta.get("errors"):
            for e in meta["errors"]:
                st.error(e)


# ---------------------------------------------------------------------------
# Summarize
# ---------------------------------------------------------------------------
def render_summarize():
    st.title("Summarize")
    vs = vector_store()
    docs = vs.list_documents()
    if not docs:
        st.warning("No documents indexed yet. Go to the Documents page first.")
        return

    doc_options = {f"{d['filename']} ({d['sha256'][:8]})": d["sha256"] for d in docs}
    selected_label = st.selectbox("Select a document to summarize", list(doc_options.keys()))
    detail_level = st.radio("Detail level", ["concise", "detailed"], horizontal=True)

    if st.button("Generate Summary", type="primary"):
        sha256 = doc_options[selected_label]
        with st.spinner("Summarizer Agent working..."):
            try:
                final_state = run_workflow(
                    user_query=f"Summarize this document ({detail_level}).",
                    model=settings.gemini_model,
                    temperature=settings.temperature,
                    top_k=settings.top_k,
                    selected_documents=[sha256],
                    mode_hint="summarizer",
                    detail_level=detail_level,
                )
            except Exception as exc:  # noqa: BLE001
                st.error(f"Summarization failed: {exc}")
                return

        result = (final_state.get("intermediate_results") or {}).get("summary_result", {})
        if final_state.get("errors"):
            for e in final_state["errors"]:
                st.error(e)
        st.subheader("Summary")
        st.markdown(result.get("summary", final_state.get("final_response", "")))
        if result.get("key_points"):
            st.subheader("Key Points")
            for kp in result["key_points"]:
                st.markdown(f"- {kp}")
        if result.get("findings"):
            st.subheader("Important Findings")
            for f in result["findings"]:
                st.markdown(f"- {f}")
        if result.get("conclusion"):
            st.subheader("Conclusion")
            st.markdown(result["conclusion"])
        _render_explainability({
            "selected_agent": final_state.get("selected_agent"),
            "routing_reason": final_state.get("routing_reason"),
            "sources": final_state.get("sources", []),
            "llm_model": final_state.get("llm_model"),
            "processing_time_seconds": final_state.get("processing_time_seconds"),
            "execution_trace": final_state.get("execution_trace", []),
        })


# ---------------------------------------------------------------------------
# Compare
# ---------------------------------------------------------------------------
def render_compare():
    st.title("Compare")
    vs = vector_store()
    docs = vs.list_documents()
    if len(docs) < 2:
        st.warning("Upload at least two documents to use the Comparator.")
        return

    doc_options = {f"{d['filename']} ({d['sha256'][:8]})": d["sha256"] for d in docs}
    labels = list(doc_options.keys())
    col1, col2 = st.columns(2)
    doc_a_label = col1.selectbox("Document A", labels, index=0)
    doc_b_label = col2.selectbox("Document B", labels, index=min(1, len(labels) - 1))

    if st.button("Generate Comparison", type="primary"):
        sha_a, sha_b = doc_options[doc_a_label], doc_options[doc_b_label]
        if sha_a == sha_b:
            st.error("Please select two different documents.")
            return
        with st.spinner("Comparator Agent working..."):
            try:
                final_state = run_workflow(
                    user_query="Compare these two documents.",
                    model=settings.gemini_model,
                    temperature=settings.temperature,
                    top_k=settings.top_k,
                    selected_documents=[sha_a, sha_b],
                    mode_hint="comparator",
                )
            except Exception as exc:  # noqa: BLE001
                st.error(f"Comparison failed: {exc}")
                return

        if final_state.get("errors"):
            for e in final_state["errors"]:
                st.error(e)
        result = (final_state.get("intermediate_results") or {}).get("comparison_result")
        if result:
            st.subheader(f"{result['document_a']}  vs  {result['document_b']}")
            if result.get("table_rows"):
                st.table(result["table_rows"])
            for section, items in result["sections"].items():
                if items:
                    st.markdown(f"**{section}**")
                    for item in items:
                        st.markdown(f"- {item}")
        else:
            st.markdown(final_state.get("final_response", ""))
        _render_explainability({
            "selected_agent": final_state.get("selected_agent"),
            "routing_reason": final_state.get("routing_reason"),
            "sources": final_state.get("sources", []),
            "llm_model": final_state.get("llm_model"),
            "processing_time_seconds": final_state.get("processing_time_seconds"),
            "execution_trace": final_state.get("execution_trace", []),
        })


# ---------------------------------------------------------------------------
# Agent Activity
# ---------------------------------------------------------------------------
def render_activity():
    st.title("Agent Activity")
    st.caption("Recent orchestrator routing / execution history (this session).")
    if not st.session_state.agent_activity:
        st.info("No activity yet. Ask a question in 'Ask Documents' to see routing decisions here.")
        return
    for entry in st.session_state.agent_activity:
        with st.container(border=True):
            st.markdown(f"**Query:** {entry['query']}")
            st.markdown(f"**Agent:** `{entry['agent']}`  |  **Time:** {entry['time']}s")
            st.caption(f"Reason: {entry['reason']}")
            if entry.get("errors"):
                for e in entry["errors"]:
                    st.error(e)


# ---------------------------------------------------------------------------
# Settings
# ---------------------------------------------------------------------------
def render_settings():
    st.title("Settings")
    st.caption("These settings affect all agents for this session.")

    from models.gemini_client import api_key_configured
    if api_key_configured():
        st.success("GEMINI_API_KEY is configured (loaded from environment / .env).")
    else:
        st.error(
            "GEMINI_API_KEY is not set. Add it to a local `.env` file (see `.env.example`) or export it "
            "as an environment variable, then restart the app. The key itself is never shown here."
        )

    settings.gemini_model = st.text_input(
        "Gemini model",
        value=settings.gemini_model,
        help="e.g. gemini-2.5-flash, gemini-2.5-pro. Must be a model available to your API key.",
    )
    settings.temperature = st.slider("Temperature", 0.0, 1.0, float(settings.temperature), 0.05)
    settings.chunk_size = st.number_input("Chunk size (characters)", min_value=200, max_value=4000,
                                           value=int(settings.chunk_size), step=50)
    settings.chunk_overlap = st.number_input("Chunk overlap (characters)", min_value=0, max_value=1000,
                                              value=int(settings.chunk_overlap), step=10)
    settings.top_k = st.number_input("Top-K retrieval", min_value=1, max_value=20, value=int(settings.top_k))

    st.info("Note: changing the embedding model requires re-indexing all documents, since existing "
            "embeddings were produced by the previous model.")
    st.text_input("Embedding model (local HuggingFace)", value=settings.embedding_model, disabled=True)

    if st.button("Save Settings"):
        st.success("Settings updated for this session.")


# ---------------------------------------------------------------------------
# Router
# ---------------------------------------------------------------------------
PAGES = {
    "Dashboard": render_dashboard,
    "Documents": render_documents,
    "Ask Documents": render_ask,
    "Summarize": render_summarize,
    "Compare": render_compare,
    "Agent Activity": render_activity,
    "Settings": render_settings,
}

PAGES[page]()
