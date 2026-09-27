"""
Comparator Agent: compares two indexed documents. Uses a raw-text diff
to find candidate changed regions, then asks the Gemini model to
semantically classify and describe the differences (added / removed /
modified / unchanged / important changes), producing a structured
comparison table.
"""
import difflib
from typing import Dict, List

from models.gemini_client import GeminiClient
from retrieval.retriever import Retriever
from utils.logger import get_logger

logger = get_logger(__name__)

COMPARE_PROMPT = """You are comparing two versions of a document for an enterprise workbench.

Only use the text provided below. Do not invent content that isn't shown.

DOCUMENT A ({name_a}):
{text_a}

DOCUMENT B ({name_b}):
{text_b}

RAW DIFF HINTS (line-level, may be noisy — use for guidance only):
{diff_hints}

Analyze the meaningful differences between the two documents (not just character-level diffs).
Return your answer strictly in this format:

## Added Content
- <item> (in B, not in A)

## Removed Content
- <item> (in A, not in B)

## Modified Content
- <item>: what changed and why it matters

## Unchanged Content
- <brief note on major unchanged sections, 1-3 bullets max>

## Important Changes
- <the most significant changes a reviewer should know about>
"""


class ComparatorAgent:
    name = "comparator"

    def __init__(self, retriever: Retriever = None, llm_client: GeminiClient = None):
        self.retriever = retriever or Retriever()
        self.llm = llm_client or GeminiClient()

    def _full_text(self, sha256: str) -> tuple[str, str]:
        chunks = self.retriever.get_full_document(sha256)
        if not chunks:
            return "", ""
        filename = chunks[0]["metadata"].get("filename", "document")
        text = "\n".join(c["text"] for c in chunks)
        return filename, text

    def _diff_hints(self, text_a: str, text_b: str, max_lines: int = 60) -> str:
        lines_a = text_a.splitlines()
        lines_b = text_b.splitlines()
        diff = list(difflib.unified_diff(lines_a, lines_b, lineterm="", n=0))
        trimmed = diff[:max_lines]
        return "\n".join(trimmed) if trimmed else "(no line-level differences detected)"

    def compare(self, sha256_a: str, sha256_b: str) -> Dict:
        if not sha256_a or not sha256_b:
            return {"error": "Two documents must be selected to run a comparison."}
        if sha256_a == sha256_b:
            return {"error": "Please select two different documents to compare."}

        name_a, text_a = self._full_text(sha256_a)
        name_b, text_b = self._full_text(sha256_b)

        if not text_a or not text_b:
            return {"error": "One or both selected documents have no indexed content."}

        diff_hints = self._diff_hints(text_a, text_b)

        prompt = COMPARE_PROMPT.format(
            name_a=name_a, text_a=text_a[:6000],
            name_b=name_b, text_b=text_b[:6000],
            diff_hints=diff_hints[:2000],
        )
        result_text = self.llm.generate(prompt)

        sections = _parse_comparison(result_text)
        table_rows = _build_table_rows(sections)

        return {
            "document_a": name_a,
            "document_b": name_b,
            "sections": sections,
            "table_rows": table_rows,
            "raw": result_text,
        }


def _parse_comparison(text: str) -> Dict[str, List[str]]:
    import re
    headers = ["Added Content", "Removed Content", "Modified Content", "Unchanged Content", "Important Changes"]
    pattern = re.split(r"##\s*(" + "|".join(headers) + r")\s*", text)
    sections = {h: [] for h in headers}
    for i in range(1, len(pattern) - 1, 2):
        header = pattern[i].strip()
        content = pattern[i + 1].strip()
        bullets = [ln.strip("-* ").strip() for ln in content.splitlines() if ln.strip().startswith(("-", "*"))]
        if header in sections:
            sections[header] = bullets or ([content] if content else [])
    return sections


def _build_table_rows(sections: Dict[str, List[str]]) -> List[Dict[str, str]]:
    rows = []
    category_map = {
        "Added Content": "Added",
        "Removed Content": "Removed",
        "Modified Content": "Modified",
        "Unchanged Content": "Unchanged",
        "Important Changes": "Important Change",
    }
    for header, category in category_map.items():
        for item in sections.get(header, []):
            rows.append({"category": category, "detail": item})
    return rows
