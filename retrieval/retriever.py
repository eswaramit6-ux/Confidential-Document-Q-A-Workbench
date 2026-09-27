"""
Retrieval helper sitting on top of VectorStore. Used by the Q&A Agent
and Comparator Agent to fetch grounded context.
"""
from typing import Dict, List, Optional

from retrieval.vector_store import VectorStore


class Retriever:
    def __init__(self, vector_store: Optional[VectorStore] = None):
        self.vector_store = vector_store or VectorStore()

    def retrieve(self, query: str, top_k: int = 4, sha256_filter: Optional[List[str]] = None) -> List[Dict]:
        if not query or not query.strip():
            return []
        return self.vector_store.query(query_text=query, top_k=top_k, sha256_filter=sha256_filter)

    def get_full_document(self, sha256: str) -> List[Dict]:
        return self.vector_store.get_document_text(sha256)
