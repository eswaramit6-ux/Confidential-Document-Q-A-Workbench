import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pytest

from models.gemini_client import GeminiUnavailableError


class FakeEmbedFn:
    """Deterministic fake embedding so tests never require downloading the real HF model."""
    def __call__(self, input):
        out = []
        for text in input:
            vec = [0.0] * 16
            for i, ch in enumerate(text[:64]):
                vec[i % 16] += ord(ch) / 1000.0
            out.append(vec)
        return out

    def name(self):
        return "fake-embed"


@pytest.fixture(autouse=True)
def _isolated_chroma_and_fake_embeddings(tmp_path, monkeypatch):
    """Every workflow test gets an isolated, empty ChromaDB directory and a fake
    (no-download) embedding function, so tests never touch real project data or
    require network access to HuggingFace."""
    import chromadb
    from models import embeddings as embeddings_module
    from retrieval import vector_store as vs_module

    monkeypatch.setattr(embeddings_module, "LocalEmbeddingFunction", lambda model_name=None: FakeEmbedFn())

    test_client = chromadb.PersistentClient(path=str(tmp_path))
    monkeypatch.setattr(vs_module, "get_chroma_client", lambda: test_client)
    monkeypatch.setattr(vs_module, "_client", None, raising=False)
    yield


class ScriptedGeminiClient:
    """Drop-in replacement for GeminiClient used only in tests (no real network/API calls)."""
    def __init__(self, route_response='{"route": "qa", "reason": "test"}', answer="Test answer.",
                 raise_on_generate=False):
        self.route_response = route_response
        self.answer = answer
        self.raise_on_generate = raise_on_generate
        self.call_count = 0
        self.model = "test-model"

    def generate(self, prompt, system=None, temperature=None):
        self.call_count += 1
        if self.raise_on_generate:
            raise GeminiUnavailableError("simulated Gemini API outage")
        if "JSON:" in prompt:
            return self.route_response
        return self.answer


def test_workflow_routes_to_qa_and_reports_not_found(monkeypatch):
    import graph.workflow as wf

    def fake_gemini_client(model=None, temperature=None):
        return ScriptedGeminiClient(route_response='{"route": "qa", "reason": "question detected"}')

    monkeypatch.setattr(wf, "GeminiClient", fake_gemini_client)

    compiled_graph, llm_client = wf.build_workflow(model="test-model", temperature=0.0, top_k=4)

    # Empty knowledge base -> QA agent should report "not found" without erroring.
    result_state = compiled_graph.invoke({
        "user_query": "What is the emergency procedure?",
        "selected_documents": [],
        "mode_hint": None,
        "retrieved_context": [],
        "intermediate_results": {},
        "sources": [],
        "errors": [],
        "execution_trace": [],
        "llm_model": "test-model",
    })

    assert result_state["selected_agent"] == "qa"
    assert "not found" in result_state["final_response"].lower()


def test_workflow_handles_gemini_outage_gracefully(monkeypatch):
    import graph.workflow as wf

    def fake_gemini_client(model=None, temperature=None):
        return ScriptedGeminiClient(raise_on_generate=True)

    monkeypatch.setattr(wf, "GeminiClient", fake_gemini_client)

    compiled_graph, llm_client = wf.build_workflow(model="test-model", temperature=0.0, top_k=4)

    result_state = compiled_graph.invoke({
        "user_query": "Summarize the safety policy.",
        "selected_documents": [],
        "mode_hint": None,
        "retrieved_context": [],
        "intermediate_results": {},
        "sources": [],
        "errors": [],
        "execution_trace": [],
        "llm_model": "test-model",
    })

    # Should not crash; orchestrator should fall back to a heuristic route, and
    # since no documents are selected, summarizer node should report a clear error, not throw.
    assert "final_response" in result_state
    assert isinstance(result_state["final_response"], str)


def test_comparator_requires_two_documents(monkeypatch):
    import graph.workflow as wf

    def fake_gemini_client(model=None, temperature=None):
        return ScriptedGeminiClient(route_response='{"route": "comparator", "reason": "compare requested"}')

    monkeypatch.setattr(wf, "GeminiClient", fake_gemini_client)
    compiled_graph, llm_client = wf.build_workflow(model="test-model", temperature=0.0, top_k=4)

    result_state = compiled_graph.invoke({
        "user_query": "Compare these documents.",
        "selected_documents": ["only_one_sha"],
        "mode_hint": "comparator",
        "retrieved_context": [],
        "intermediate_results": {},
        "sources": [],
        "errors": [],
        "execution_trace": [],
        "llm_model": "test-model",
    })

    assert result_state["selected_agent"] == "comparator"
    assert result_state["errors"]
    assert "two documents" in result_state["final_response"].lower()


def test_orchestrator_reports_missing_api_key_clearly():
    """When GEMINI_API_KEY is missing, the real GeminiClient should raise
    GeminiAPIKeyMissingError, and the Orchestrator should fall back to its
    safe heuristic route rather than crashing, while still surfacing the
    root cause in the routing decision."""
    from agents.orchestrator import Orchestrator
    from models.gemini_client import GeminiClient

    client = GeminiClient(model="test-model", temperature=0.0, api_key="")
    orch = Orchestrator(llm_client=client)

    decision = orch.route("What is the emergency procedure?")

    assert decision.used_fallback is True
    assert "GEMINI_API_KEY" in decision.raw_model_output


def test_workflow_surfaces_missing_api_key_when_llm_is_actually_invoked(monkeypatch):
    """With real indexed content (so the Q&A agent actually needs to call the
    LLM to generate an answer) and no GEMINI_API_KEY configured, the workflow
    should surface a clear configuration error instead of crashing."""
    import graph.workflow as wf
    from models.gemini_client import GeminiClient
    from retrieval.vector_store import VectorStore
    from ingestion.chunker import Chunk

    monkeypatch.setattr(wf, "GeminiClient",
                         lambda model=None, temperature=None: GeminiClient(model=model, temperature=temperature,
                                                                            api_key=""))

    # Seed the isolated test ChromaDB with one real chunk so retrieval succeeds
    # and the QA agent proceeds to call the (misconfigured) LLM.
    store = VectorStore()
    store.add_chunks([
        Chunk(text="Press the red button to initiate emergency shutdown.",
              chunk_index=0, page_number=12, filename="SOP.pdf", sha256="sopsha")
    ])

    compiled_graph, llm_client = wf.build_workflow(model="test-model", temperature=0.0, top_k=4)

    result_state = compiled_graph.invoke({
        "user_query": "What is the emergency procedure?",
        "selected_documents": [],
        "mode_hint": "qa",
        "retrieved_context": [],
        "intermediate_results": {},
        "sources": [],
        "errors": [],
        "execution_trace": [],
        "llm_model": "test-model",
    })

    assert result_state["errors"]
    assert any("GEMINI_API_KEY" in e for e in result_state["errors"])
    assert "gemini api key" in result_state["final_response"].lower()
