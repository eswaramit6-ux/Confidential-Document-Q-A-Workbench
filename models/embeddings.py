"""
Local embedding model wrapper using HuggingFace sentence-transformers
(all-MiniLM-L6-v2). Runs fully on-device — no network calls except the
one-time HuggingFace model download the first time it's used (standard
for local HF models; after that it is cached locally and works offline).
"""
from functools import lru_cache
from typing import List

from utils.config import DEFAULT_EMBEDDING_MODEL
from utils.logger import get_logger

logger = get_logger(__name__)

_model_cache = {}


def _load_model(model_name: str):
    if model_name in _model_cache:
        return _model_cache[model_name]
    from sentence_transformers import SentenceTransformer
    logger.info(f"Loading local embedding model: {model_name}")
    model = SentenceTransformer(model_name)
    _model_cache[model_name] = model
    return model


class LocalEmbeddingFunction:
    """Chroma-compatible embedding function backed by a local SentenceTransformer."""

    def __init__(self, model_name: str = DEFAULT_EMBEDDING_MODEL):
        self.model_name = model_name

    def __call__(self, input: List[str]) -> List[List[float]]:
        model = _load_model(self.model_name)
        embeddings = model.encode(list(input), show_progress_bar=False, convert_to_numpy=True)
        return embeddings.tolist()

    def name(self) -> str:
        return f"local-hf::{self.model_name}"


def embedding_model_available(model_name: str = DEFAULT_EMBEDDING_MODEL) -> tuple[bool, str]:
    """Health check: attempt to load the embedding model and embed a trivial string."""
    try:
        model = _load_model(model_name)
        _ = model.encode(["healthcheck"], show_progress_bar=False)
        return True, "Embedding model available."
    except Exception as exc:  # noqa: BLE001
        return False, f"Embedding model unavailable: {exc}"
