"""
Summarizer Agent: chunk-level map summarization followed by a final
combined (reduce) summary, plus key points / findings / conclusion
extraction. Grounded strictly in the retrieved chunk text — the prompt
explicitly instructs the model not to invent information. LLM calls are
made through the Gemini API (see models/gemini_client.py).
"""
from typing import Dict, List

from models.gemini_client import GeminiClient
from retrieval.retriever import Retriever
from utils.logger import get_logger

logger = get_logger(__name__)

NO_HALLUCINATION_RULE = (
    "You must only use information explicitly present in the provided document text. "
    "Do not invent facts, numbers, names, or claims that are not present in the text. "
    "If the text is insufficient to answer part of the request, say so plainly."
)

MAP_PROMPT = """{rule}

Summarize the following document excerpt in 3-5 concise sentences, capturing only the concrete
information present in the text.

EXCERPT:
{chunk}

SUMMARY:"""

REDUCE_PROMPT = """{rule}

You are given several partial summaries of sequential sections of ONE document. Combine them into a
single coherent summary. Detail level requested: {detail_level}.

PARTIAL SUMMARIES:
{summaries}

Return your answer using this exact structure:

## Summary
<a coherent {detail_level} summary of the whole document>

## Key Points
- <bullet 1>
- <bullet 2>
...

## Important Findings
- <finding 1>
...

## Conclusion
<one short paragraph conclusion drawn only from the material above>
"""


class SummarizerAgent:
    name = "summarizer"

    def __init__(self, retriever: Retriever = None, llm_client: GeminiClient = None):
        self.retriever = retriever or Retriever()
        self.llm = llm_client or GeminiClient()

    def summarize_document(self, sha256: str, detail_level: str = "concise", max_chunks_per_batch: int = 6) -> Dict:
        chunks = self.retriever.get_full_document(sha256)
        if not chunks:
            return {
                "summary": "No indexed content was found for this document.",
                "key_points": [],
                "findings": [],
                "conclusion": "",
                "chunks_used": 0,
            }

        filename = chunks[0]["metadata"].get("filename", "document")

        # Map step: summarize batches of chunks to keep prompts bounded.
        partial_summaries: List[str] = []
        batch: List[str] = []
        for i, c in enumerate(chunks):
            batch.append(c["text"])
            if len(batch) >= max_chunks_per_batch or i == len(chunks) - 1:
                joined = "\n\n".join(batch)
                prompt = MAP_PROMPT.format(rule=NO_HALLUCINATION_RULE, chunk=joined[:6000])
                try:
                    partial = self.llm.generate(prompt)
                except Exception as exc:  # noqa: BLE001
                    logger.error(f"Summarizer map step failed: {exc}")
                    raise
                partial_summaries.append(partial)
                batch = []

        # Reduce step: combine partial summaries into the final structured summary.
        detail_label = "detailed, multi-paragraph" if detail_level == "detailed" else "concise, high-level"
        reduce_prompt = REDUCE_PROMPT.format(
            rule=NO_HALLUCINATION_RULE,
            detail_level=detail_label,
            summaries="\n\n---\n\n".join(f"[Section {i+1}] {s}" for i, s in enumerate(partial_summaries)),
        )
        final_text = self.llm.generate(reduce_prompt)

        key_points, findings, conclusion, summary_body = _parse_structured_summary(final_text)

        return {
            "filename": filename,
            "summary": summary_body or final_text,
            "key_points": key_points,
            "findings": findings,
            "conclusion": conclusion,
            "chunks_used": len(chunks),
            "raw": final_text,
        }


def _parse_structured_summary(text: str):
    """Best-effort parsing of the '## Summary/## Key Points/## Important Findings/## Conclusion' structure."""
    import re
    sections = {"summary": "", "key_points": "", "findings": "", "conclusion": ""}
    pattern = re.split(r"##\s*(Summary|Key Points|Important Findings|Conclusion)\s*", text, flags=re.IGNORECASE)
    # pattern like: [preamble, 'Summary', content, 'Key Points', content, ...]
    for i in range(1, len(pattern) - 1, 2):
        header = pattern[i].strip().lower()
        content = pattern[i + 1].strip()
        if header.startswith("summary"):
            sections["summary"] = content
        elif header.startswith("key points"):
            sections["key_points"] = content
        elif header.startswith("important findings"):
            sections["findings"] = content
        elif header.startswith("conclusion"):
            sections["conclusion"] = content

    def to_bullets(block: str):
        return [ln.strip("-* ").strip() for ln in block.splitlines() if ln.strip().startswith(("-", "*"))]

    key_points = to_bullets(sections["key_points"])
    findings = to_bullets(sections["findings"])
    return key_points, findings, sections["conclusion"], sections["summary"]
