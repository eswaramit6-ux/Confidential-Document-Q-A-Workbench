import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pytest


class FakeEmbedFn:
    """Deterministic fake embeddings so tests don't need to download a real HF model."""
    def __call__(self, input):
        out = []
        for text in input:
            # simple hashed bag-of-words -> fixed-size vector, deterministic
            vec = [0.0] * 16
            for i, ch in enumerate(text[:64]):
                vec[i % 16] += ord(ch) / 1000.0
            out.append(vec)
        return out

    def name(self):
        return "fake-embed"


@pytest.fixture
def vector_store(tmp_path, monkeypatch):
    from retrieval import vector_store as vs_module
    monkeypatch.setattr(vs_module, "CHROMA_DIR", str(tmp_path), raising=False)
    import chromadb
    client = chromadb.PersistentClient(path=str(tmp_path))
    monkeypatch.setattr(vs_module, "get_chroma_client", lambda: client)

    store = vs_module.VectorStore.__new__(vs_module.VectorStore)
    store.client = client
    store.embed_fn = FakeEmbedFn()
    store.collection = client.get_or_create_collection(name="test_collection")
    return store


def _fake_chunk(text, idx, sha, filename="doc.pdf", page=1):
    from ingestion.chunker import Chunk
    return Chunk(text=text, chunk_index=idx, page_number=page, filename=filename, sha256=sha)


def test_add_and_query_chunks(vector_store):
    sha = "abc123"
    chunks = [
        _fake_chunk("The emergency shutdown procedure requires pressing the red button.", 0, sha),
        _fake_chunk("Regular maintenance should occur every 6 months.", 1, sha),
    ]
    added = vector_store.add_chunks(chunks)
    assert added == 2
    assert vector_store.count_chunks() == 2

    results = vector_store.query("emergency shutdown procedure", top_k=2)
    assert len(results) == 2
    combined = " ".join(r["text"].lower() for r in results)
    assert "shutdown" in combined
    assert all("metadata" in r and "distance" in r for r in results)


def test_duplicate_detection(vector_store):
    sha = "dupsha"
    chunks = [_fake_chunk("some content", 0, sha)]
    vector_store.add_chunks(chunks)
    assert vector_store.document_exists(sha) is True
    assert vector_store.document_exists("nonexistent") is False


def test_delete_document(vector_store):
    sha = "delsha"
    chunks = [_fake_chunk("content to delete", 0, sha), _fake_chunk("more content", 1, sha)]
    vector_store.add_chunks(chunks)
    removed = vector_store.delete_document(sha)
    assert removed == 2
    assert vector_store.count_chunks() == 0


def test_list_documents(vector_store):
    vector_store.add_chunks([_fake_chunk("a", 0, "sha1", filename="a.pdf")])
    vector_store.add_chunks([_fake_chunk("b", 0, "sha2", filename="b.pdf")])
    docs = vector_store.list_documents()
    filenames = {d["filename"] for d in docs}
    assert filenames == {"a.pdf", "b.pdf"}
