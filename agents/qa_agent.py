"""
Q&A Agent: retrieval-augmented generation grounded strictly in
retrieved chunks from ChromaDB. Always cites filename + page number
(where available) and explicitly states when the answer is not found.
Retrieval and embeddings remain fully local; only the final answer
generation step calls the Gemini API.
"""
from typing import Dict, List, Optional

from models.gemini_client import GeminiClient
from retrieval.retriever import Retriever
from utils.logger import get_logger

logger = get_logger(__name__)

QA_PROMPT = """You are a grounded document Q&A assistant for confidential internal documents.

RULES:
- Answer ONLY using the CONTEXT below.
- Never invent information that is not present in the context.
- If the context does not contain the answer, respond exactly with:
  "The answer was not found in the indexed documents."
- Be concise and factual.

CONTEXT:
{context}

QUESTION:
{question}

ANSWER:"""


class QAAgent:
    name = "qa"
    NOT_FOUND_MESSAGE = "The answer was not found in the indexed documents."

    def __init__(self, retriever: Retriever = None, llm_client: GeminiClient = None):
        self.retriever = retriever or Retriever()
        self.llm = llm_client or GeminiClient()

    def answer(self, question: str, top_k: int = 4, sha256_filter: Optional[List[str]] = None) -> Dict:
        if not question or not question.strip():
            return {
                "answer": "Please enter a question.",
                "sources": [],
                "retrieved_chunks": [],
            }

        retrieved = self.retriever.retrieve(question, top_k=top_k, sha256_filter=sha256_filter)

        if not retrieved:
            return {
                "answer": self.NOT_FOUND_MESSAGE,
                "sources": [],
                "retrieved_chunks": [],
            }

        context_blocks = []
        for r in retrieved:
            meta = r["metadata"]
            page = meta.get("page_number", 0)
            page_str = f"Page {page}" if page and page > 0 else "N/A"
            context_blocks.append(f"[{meta.get('filename')} — {page_str}]\n{r['text']}")
        context = "\n\n---\n\n".join(context_blocks)

        prompt = QA_PROMPT.format(context=context[:8000], question=question)
        answer_text = self.llm.generate(prompt)

        sources = []
        seen = set()
        for r in retrieved:
            meta = r["metadata"]
            key = (meta.get("filename"), meta.get("page_number"))
            if key in seen:
                continue
            seen.add(key)
            sources.append({
                "filename": meta.get("filename"),
                "page_number": meta.get("page_number") if meta.get("page_number") else None,
            })

        return {
            "answer": answer_text,
            "sources": sources,
            "retrieved_chunks": retrieved,
        }
