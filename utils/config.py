"""
Central configuration for the Confidential Document Q&A Workbench.

Document storage, chunking, embeddings, and ChromaDB remain fully local.
LLM inference (routing, summarization, Q&A, comparison) is provided by the
Google Gemini API and therefore requires outbound network access and a
configured GEMINI_API_KEY. This application is NOT fully offline.
"""
import os
from dataclasses import dataclass, field

from dotenv import load_dotenv

# Load a local .env file (if present) into the environment before anything else reads it.
load_dotenv()

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, "data")
UPLOADS_DIR = os.path.join(DATA_DIR, "uploads")
CHROMA_DIR = os.path.join(DATA_DIR, "chroma")

os.makedirs(UPLOADS_DIR, exist_ok=True)
os.makedirs(CHROMA_DIR, exist_ok=True)

DEFAULT_GEMINI_MODEL = "gemini-2.5-flash"
DEFAULT_EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"  # kept local, never sent to Gemini
CHROMA_COLLECTION_NAME = "confidential_documents"

DEFAULT_CHUNK_SIZE = 1000
DEFAULT_CHUNK_OVERLAP = 150
DEFAULT_TOP_K = 4
DEFAULT_TEMPERATURE = 0.1

SUPPORTED_EXTENSIONS = (".pdf", ".docx")


@dataclass
class RuntimeSettings:
    """Mutable runtime settings, editable from the Streamlit Settings page."""
    gemini_model: str = DEFAULT_GEMINI_MODEL
    temperature: float = DEFAULT_TEMPERATURE
    chunk_size: int = DEFAULT_CHUNK_SIZE
    chunk_overlap: int = DEFAULT_CHUNK_OVERLAP
    top_k: int = DEFAULT_TOP_K
    embedding_model: str = DEFAULT_EMBEDDING_MODEL


def get_settings() -> RuntimeSettings:
    """Fetch (or lazily create) the shared runtime settings object from Streamlit session state."""
    import streamlit as st
    if "runtime_settings" not in st.session_state:
        st.session_state.runtime_settings = RuntimeSettings()
    return st.session_state.runtime_settings
