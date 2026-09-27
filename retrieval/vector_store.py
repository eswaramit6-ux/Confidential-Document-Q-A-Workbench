"""
Persistent local ChromaDB vector store. All embeddings, storage, and
querying happen on-disk in ./data/chroma — nothing is sent to a cloud
vector database.
"""
from typing import Dict, List, Optional

import chromadb

from utils.config import CHROMA_COLLECTION_NAME, CHROMA_DIR, DEFAULT_EMBEDDING_MODEL
from utils.logger import get_logger

logger = get_logger(__name__)

_client = None


def get_chroma_client():
    global _client
    if _client is None:
        _client = chromadb.PersistentClient(path=CHROMA_DIR)
    return _client


class VectorStore:
    def __init__(self, embedding_model_name: str = DEFAULT_EMBEDDING_MODEL,
                 collection_name: str = CHROMA_COLLECTION_NAME):
        from models.embeddings import LocalEmbeddingFunction
        self.client = get_chroma_client()
        self.embed_fn = LocalEmbeddingFunction(embedding_model_name)
        self.collection = self.client.get_or_create_collection(
            name=collection_name,
            metadata={"hnsw:space": "cosine"},
        )

    # ---------- Document-level operations ----------

    def document_exists(self, sha256: str) -> bool:
        try:
            result = self.collection.get(where={"sha256": sha256}, limit=1)
            return bool(result and result.get("ids"))
        except Exception:  # noqa: BLE001
            return False

    def list_documents(self) -> List[Dict]:
        """Return one summary row per unique document (by sha256)."""
        try:
            result = self.collection.get(include=["metadatas"])
        except Exception:  # noqa: BLE001
            return []
        metadatas = result.get("metadatas", []) or []
        docs: Dict[str, Dict] = {}
        for meta in metadatas:
            if not meta:
                continue
            sha = meta.get("sha256")
            if sha not in docs:
                docs[sha] = {
                    "filename": meta.get("filename"),
                    "sha256": sha,
                    "chunk_count": 0,
                    "has_pages": meta.get("page_number", 0) > 0,
                }
            docs[sha]["chunk_count"] += 1
        return list(docs.values())

    def delete_document(self, sha256: str) -> int:
        result = self.collection.get(where={"sha256": sha256})
        ids = result.get("ids", []) or []
        if ids:
            self.collection.delete(ids=ids)
        return len(ids)

    def count_chunks(self) -> int:
        try:
            return self.collection.count()
        except Exception:  # noqa: BLE001
            return 0

    # ---------- Indexing ----------

    def add_chunks(self, chunks: List) -> int:
        if not chunks:
            return 0
        ids = [f"{c.sha256}_{c.chunk_index}" for c in chunks]
        documents = [c.text for c in chunks]
        metadatas = [
            {
                "filename": c.filename,
                "sha256": c.sha256,
                "chunk_index": c.chunk_index,
                "page_number": c.page_number,
            }
            for c in chunks
        ]
        embeddings = self.embed_fn(documents)
        self.collection.add(ids=ids, documents=documents, metadatas=metadatas, embeddings=embeddings)
        return len(ids)

    # ---------- Retrieval ----------

    def query(self, query_text: str, top_k: int = 4, sha256_filter: Optional[List[str]] = None) -> List[Dict]:
        query_embedding = self.embed_fn([query_text])[0]
        where = None
        if sha256_filter:
            where = {"sha256": {"$in": sha256_filter}} if len(sha256_filter) > 1 else {"sha256": sha256_filter[0]}
        results = self.collection.query(
            query_embeddings=[query_embedding],
            n_results=top_k,
            where=where,
            include=["documents", "metadatas", "distances"],
        )
        output = []
        docs = results.get("documents", [[]])[0]
        metas = results.get("metadatas", [[]])[0]
        dists = results.get("distances", [[]])[0]
        for text, meta, dist in zip(docs, metas, dists):
            output.append({"text": text, "metadata": meta, "distance": dist})
        return output

    def get_document_text(self, sha256: str) -> List[Dict]:
        """Return all chunks for a document ordered by chunk_index (for summarization/comparison)."""
        result = self.collection.get(where={"sha256": sha256}, include=["documents", "metadatas"])
        rows = list(zip(result.get("documents", []), result.get("metadatas", [])))
        rows.sort(key=lambda r: r[1].get("chunk_index", 0))
        return [{"text": t, "metadata": m} for t, m in rows]


def health_check() -> tuple[bool, str]:
    try:
        client = get_chroma_client()
        client.heartbeat()
        return True, "ChromaDB is available."
    except Exception as exc:  # noqa: BLE001
        return False, f"ChromaDB unavailable: {exc}"
