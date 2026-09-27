"""
Text chunking with configurable size/overlap. Preserves page-number
metadata (where available) so retrieved chunks can be cited by page.
"""
from dataclasses import dataclass
from typing import List

from ingestion.document_loader import ExtractedDocument
from utils.config import DEFAULT_CHUNK_OVERLAP, DEFAULT_CHUNK_SIZE


@dataclass
class Chunk:
    text: str
    chunk_index: int
    page_number: int  # 0 means "not applicable" (e.g. DOCX)
    filename: str
    sha256: str


def _split_text(text: str, chunk_size: int, overlap: int) -> List[str]:
    text = text.strip()
    if not text:
        return []
    if overlap >= chunk_size:
        overlap = max(0, chunk_size // 4)
    chunks = []
    start = 0
    length = len(text)
    while start < length:
        end = min(start + chunk_size, length)
        # try to break on a sentence/paragraph boundary near the end
        segment = text[start:end]
        if end < length:
            last_break = max(segment.rfind(". "), segment.rfind("\n"))
            if last_break > chunk_size * 0.5:
                end = start + last_break + 1
                segment = text[start:end]
        chunks.append(segment.strip())
        if end >= length:
            break
        start = max(end - overlap, start + 1)
    return [c for c in chunks if c]


def chunk_document(
    doc: ExtractedDocument,
    chunk_size: int = DEFAULT_CHUNK_SIZE,
    overlap: int = DEFAULT_CHUNK_OVERLAP,
) -> List[Chunk]:
    all_chunks: List[Chunk] = []
    idx = 0
    for page in doc.pages:
        pieces = _split_text(page.text, chunk_size, overlap)
        for piece in pieces:
            all_chunks.append(
                Chunk(
                    text=piece,
                    chunk_index=idx,
                    page_number=page.page_number,
                    filename=doc.filename,
                    sha256=doc.sha256,
                )
            )
            idx += 1
    return all_chunks
