import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from agents.orchestrator import Orchestrator, RoutingDecision
from agents.qa_agent import QAAgent
from models.gemini_client import GeminiUnavailableError


class FakeLLM:
    """Fake Gemini-compatible client for deterministic, offline testing."""
    def __init__(self, scripted_response: str = "", raise_error: bool = False):
        self.scripted_response = scripted_response
        self.raise_error = raise_error
        self.calls = []

    def generate(self, prompt, system=None, temperature=None):
        self.calls.append(prompt)
        if self.raise_error:
            raise GeminiUnavailableError("Gemini API is not reachable (simulated).")
        return self.scripted_response


class FakeRetriever:
    def __init__(self, results=None):
        self._results = results or []

    def retrieve(self, query, top_k=4, sha256_filter=None):
        if not query.strip():
            return []
        return self._results

    def get_full_document(self, sha256):
        return self._results


def test_orchestrator_parses_valid_json_route():
    llm = FakeLLM(scripted_response='{"route": "qa", "reason": "The user asked a specific question."}')
    orch = Orchestrator(llm_client=llm)
    decision = orch.route("What is the emergency shutdown procedure?")
    assert decision.route == "qa"
    assert not decision.used_fallback


def test_orchestrator_handles_comparator_intent():
    llm = FakeLLM(scripted_response='{"route": "comparator", "reason": "Comparing two SOP versions."}')
    orch = Orchestrator(llm_client=llm)
    decision = orch.route("What changed between SOP version 1 and version 2?")
    assert decision.route == "comparator"


def test_orchestrator_falls_back_on_malformed_json():
    llm = FakeLLM(scripted_response="not valid json at all, sorry")
    orch = Orchestrator(llm_client=llm)
    decision = orch.route("Summarize the safety policy.")
    assert decision.used_fallback is True
    assert decision.route in {"summarizer", "qa", "comparator", "multi_agent", "unsupported"}


def test_orchestrator_falls_back_when_llm_unreachable():
    llm = FakeLLM(raise_error=True)
    orch = Orchestrator(llm_client=llm)
    decision = orch.route("Compare both SOPs and summarize the important changes.")
    assert decision.used_fallback is True
    # keyword heuristic should catch "compare"
    assert decision.route == "comparator"


def test_orchestrator_rejects_invalid_route_value():
    llm = FakeLLM(scripted_response='{"route": "make_coffee", "reason": "n/a"}')
    orch = Orchestrator(llm_client=llm)
    decision = orch.route("Do something unrelated.")
    assert decision.used_fallback is True


def test_qa_agent_answers_when_context_found():
    retriever = FakeRetriever(results=[
        {"text": "Press the red button to initiate emergency shutdown.",
         "metadata": {"filename": "SOP.pdf", "page_number": 12}, "distance": 0.1}
    ])
    llm = FakeLLM(scripted_response="Press the red button to initiate emergency shutdown.")
    agent = QAAgent(retriever=retriever, llm_client=llm)
    result = agent.answer("What is the emergency shutdown procedure?")
    assert "red button" in result["answer"].lower()
    assert result["sources"][0]["filename"] == "SOP.pdf"
    assert result["sources"][0]["page_number"] == 12


def test_qa_agent_reports_not_found_when_no_chunks_retrieved():
    retriever = FakeRetriever(results=[])
    llm = FakeLLM(scripted_response="should not be called")
    agent = QAAgent(retriever=retriever, llm_client=llm)
    result = agent.answer("What is the meaning of life?")
    assert result["answer"] == QAAgent.NOT_FOUND_MESSAGE
    assert result["sources"] == []


def test_qa_agent_handles_empty_question():
    agent = QAAgent(retriever=FakeRetriever(), llm_client=FakeLLM())
    result = agent.answer("")
    assert "enter a question" in result["answer"].lower()
