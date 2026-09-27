"""
Orchestrates: raw file bytes -> text extraction -> chunking -> local
embeddings -> ChromaDB. Includes SHA-256 duplicate detection so the
same document is never re-embedded unnecessarily.
"""
import os
from dataclasses import dataclass
from typing import Optional

from ingestion.chunker import chunk_document
from ingestion.document_loader import (
    CorruptedDocumentError,
    EmptyDocumentError,
    UnsupportedFileTypeError,
    compute_sha256,
    extract_document,
)
from retrieval.vector_store import VectorStore
from utils.config import DEFAULT_CHUNK_OVERLAP, DEFAULT_CHUNK_SIZE, UPLOADS_DIR
from utils.logger import get_logger, safe_snippet

logger = get_logger(__name__)


@dataclass
class IndexResult:
    filename: str
    sha256: str
    status: str  # "indexed" | "duplicate" | "error"
    chunk_count: int = 0
    message: str = ""


def index_file(
    file_bytes: bytes,
    filename: str,
    vector_store: Optional[VectorStore] = None,
    chunk_size: int = DEFAULT_CHUNK_SIZE,
    chunk_overlap: int = DEFAULT_CHUNK_OVERLAP,
    force_reindex: bool = False,
) -> IndexResult:
    vector_store = vector_store or VectorStore()

    if not filename.lower().endswith((".pdf", ".docx")):
        return IndexResult(filename=filename, sha256="", status="error",
                            message="Unsupported file type. Only PDF and DOCX are allowed.")

    sha256 = compute_sha256(file_bytes)

    if not force_reindex and vector_store.document_exists(sha256):
        logger.info(f"Duplicate detected for '{filename}' (sha256={sha256[:12]}...)")
        return IndexResult(filename=filename, sha256=sha256, status="duplicate",
                            message=f"'{filename}' is already indexed (duplicate content detected).")

    try:
        extracted = extract_document(file_bytes, filename)
    except (UnsupportedFileTypeError, CorruptedDocumentError, EmptyDocumentError) as exc:
        logger.warning(f"Ingestion failed for '{filename}': {exc}")
        return IndexResult(filename=filename, sha256=sha256, status="error", message=str(exc))
    except Exception as exc:  # noqa: BLE001
        logger.error(f"Unexpected ingestion error for '{filename}': {exc}")
        return IndexResult(filename=filename, sha256=sha256, status="error",
                            message=f"Unexpected error while processing '{filename}'.")

    if force_reindex:
        removed = vector_store.delete_document(sha256)
        if removed:
            logger.info(f"Re-index: removed {removed} old chunks for '{filename}'")

    chunks = chunk_document(extracted, chunk_size=chunk_size, overlap=chunk_overlap)
    if not chunks:
        return IndexResult(filename=filename, sha256=sha256, status="error",
                            message=f"'{filename}' produced no usable text chunks.")

    try:
        added = vector_store.add_chunks(chunks)
    except Exception as exc:  # noqa: BLE001
        logger.error(f"Embedding/indexing failed for '{filename}': {exc}")
        return IndexResult(filename=filename, sha256=sha256, status="error",
                            message=f"Failed to embed/index '{filename}': {exc}")

    # Persist a local copy of the raw file (kept local; never uploaded elsewhere)
    try:
        os.makedirs(UPLOADS_DIR, exist_ok=True)
        safe_name = f"{sha256[:16]}_{filename}"
        with open(os.path.join(UPLOADS_DIR, safe_name), "wb") as f:
            f.write(file_bytes)
    except OSError as exc:
        logger.warning(f"Could not persist raw file copy for '{filename}': {exc}")

    logger.info(f"Indexed '{filename}' -> {added} chunks | preview: {safe_snippet(extracted.full_text)}")
    return IndexResult(filename=filename, sha256=sha256, status="indexed", chunk_count=added,
                        message=f"'{filename}' indexed successfully ({added} chunks).")
