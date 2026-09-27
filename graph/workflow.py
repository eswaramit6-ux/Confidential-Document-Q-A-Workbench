"""
LangGraph workflow wiring the Orchestrator and the three specialist
agents together through the shared WorkbenchState.

    User -> Orchestrator -> {summarizer | qa | comparator | multi_agent} -> Final Response

LLM inference (routing + all three specialist agents) is provided by the
Google Gemini API (models/gemini_client.py). Retrieval, embeddings, and
ChromaDB storage remain fully local.
"""
import time
from typing import Optional

from langgraph.graph import END, StateGraph

from agents.comparator import ComparatorAgent
from agents.orchestrator import Orchestrator
from agents.qa_agent import QAAgent
from agents.summarizer import SummarizerAgent
from graph.state import WorkbenchState
from models.gemini_client import GeminiAPIKeyMissingError, GeminiClient, GeminiUnavailableError
from retrieval.retriever import Retriever
from utils.logger import get_logger

logger = get_logger(__name__)


def build_workflow(model: str, temperature: float, top_k: int):
    """Construct a fresh LangGraph workflow bound to the given runtime settings."""
    llm_client = GeminiClient(model=model, temperature=temperature)
    retriever = Retriever()
    orchestrator = Orchestrator(llm_client=llm_client)
    summarizer_agent = SummarizerAgent(retriever=retriever, llm_client=llm_client)
    qa_agent = QAAgent(retriever=retriever, llm_client=llm_client)
    comparator_agent = ComparatorAgent(retriever=retriever, llm_client=llm_client)

    def orchestrator_node(state: WorkbenchState) -> WorkbenchState:
        trace = state.get("execution_trace", [])
        trace.append("Orchestrator: classifying user intent via Gemini")
        try:
            decision = orchestrator.route(state.get("user_query", ""), mode_hint=state.get("mode_hint"))
            state["selected_agent"] = decision.route
            state["routing_reason"] = decision.reason
            if decision.used_fallback:
                trace.append(f"Orchestrator: used safe fallback heuristic -> route='{decision.route}'")
            else:
                trace.append(f"Orchestrator: LLM selected route='{decision.route}'")
        except Exception as exc:  # noqa: BLE001
            logger.error(f"Orchestrator node failed: {exc}")
            state["selected_agent"] = "unsupported"
            state["routing_reason"] = "Routing failed; see errors."
            state.setdefault("errors", []).append(f"Orchestrator error: {exc}")
        state["execution_trace"] = trace
        return state

    def summarizer_node(state: WorkbenchState) -> WorkbenchState:
        trace = state.get("execution_trace", [])
        trace.append("Summarizer Agent: generating chunk-level + combined summary")
        docs = state.get("selected_documents") or []
        if not docs:
            state.setdefault("errors", []).append("No document selected for summarization.")
            state["final_response"] = "Please select a document to summarize."
            state["execution_trace"] = trace
            return state
        try:
            detail = (state.get("intermediate_results") or {}).get("detail_level", "concise")
            result = summarizer_agent.summarize_document(docs[0], detail_level=detail)
            state.setdefault("intermediate_results", {})["summary_result"] = result
            state["final_response"] = result.get("summary", "")
            state["sources"] = [{"filename": result.get("filename"), "page_number": None}]
        except GeminiAPIKeyMissingError as exc:
            state.setdefault("errors", []).append(str(exc))
            state["final_response"] = f"Gemini API key not configured: {exc}"
        except GeminiUnavailableError as exc:
            state.setdefault("errors", []).append(str(exc))
            state["final_response"] = f"Gemini API error: {exc}"
        except Exception as exc:  # noqa: BLE001
            logger.error(f"Summarizer node failed: {exc}")
            state.setdefault("errors", []).append(f"Summarizer error: {exc}")
            state["final_response"] = "The Summarizer Agent encountered an error processing this document."
        state["execution_trace"] = trace
        return state

    def qa_node(state: WorkbenchState) -> WorkbenchState:
        trace = state.get("execution_trace", [])
        trace.append("Q&A Agent: retrieving relevant chunks from ChromaDB")
        try:
            docs = state.get("selected_documents") or None
            result = qa_agent.answer(state.get("user_query", ""), top_k=top_k, sha256_filter=docs)
            trace.append(f"Q&A Agent: retrieved {len(result.get('retrieved_chunks', []))} chunk(s)")
            trace.append("Q&A Agent: generating grounded answer via Gemini")
            state["final_response"] = result["answer"]
            state["sources"] = result["sources"]
            state["retrieved_context"] = result["retrieved_chunks"]
        except GeminiAPIKeyMissingError as exc:
            state.setdefault("errors", []).append(str(exc))
            state["final_response"] = f"Gemini API key not configured: {exc}"
        except GeminiUnavailableError as exc:
            state.setdefault("errors", []).append(str(exc))
            state["final_response"] = f"Gemini API error: {exc}"
        except Exception as exc:  # noqa: BLE001
            logger.error(f"QA node failed: {exc}")
            state.setdefault("errors", []).append(f"Q&A error: {exc}")
            state["final_response"] = "The Q&A Agent encountered an error answering this question."
        state["execution_trace"] = trace
        return state

    def comparator_node(state: WorkbenchState) -> WorkbenchState:
        trace = state.get("execution_trace", [])
        trace.append("Comparator Agent: diffing and semantically analyzing two documents")
        docs = state.get("selected_documents") or []
        if len(docs) < 2:
            state.setdefault("errors", []).append("Two documents must be selected for comparison.")
            state["final_response"] = "Please select two documents to compare."
            state["execution_trace"] = trace
            return state
        try:
            result = comparator_agent.compare(docs[0], docs[1])
            if "error" in result:
                state.setdefault("errors", []).append(result["error"])
                state["final_response"] = result["error"]
            else:
                state.setdefault("intermediate_results", {})["comparison_result"] = result
                state["final_response"] = result["raw"]
                state["sources"] = [
                    {"filename": result["document_a"], "page_number": None},
                    {"filename": result["document_b"], "page_number": None},
                ]
        except GeminiAPIKeyMissingError as exc:
            state.setdefault("errors", []).append(str(exc))
            state["final_response"] = f"Gemini API key not configured: {exc}"
        except GeminiUnavailableError as exc:
            state.setdefault("errors", []).append(str(exc))
            state["final_response"] = f"Gemini API error: {exc}"
        except Exception as exc:  # noqa: BLE001
            logger.error(f"Comparator node failed: {exc}")
            state.setdefault("errors", []).append(f"Comparator error: {exc}")
            state["final_response"] = "The Comparator Agent encountered an error comparing these documents."
        state["execution_trace"] = trace
        return state

    def multi_agent_node(state: WorkbenchState) -> WorkbenchState:
        """Handles requests needing comparator + summarizer-style synthesis in one pass."""
        trace = state.get("execution_trace", [])
        trace.append("Multi-Agent: running Comparator then synthesizing a summarized view of the changes")
        docs = state.get("selected_documents") or []
        if len(docs) < 2:
            state.setdefault("errors", []).append("Two documents must be selected for multi-agent comparison+summary.")
            state["final_response"] = "Please select two documents for this combined request."
            state["execution_trace"] = trace
            return state
        try:
            comparison = comparator_agent.compare(docs[0], docs[1])
            if "error" in comparison:
                state.setdefault("errors", []).append(comparison["error"])
                state["final_response"] = comparison["error"]
                state["execution_trace"] = trace
                return state
            important = comparison["sections"].get("Important Changes", [])
            summary_text = (
                "## Combined Comparison + Summary\n\n"
                f"**Documents compared:** {comparison['document_a']} vs {comparison['document_b']}\n\n"
                "### Most Important Changes\n" + "\n".join(f"- {i}" for i in important) +
                "\n\n### Full Structured Comparison\n" + comparison["raw"]
            )
            state.setdefault("intermediate_results", {})["comparison_result"] = comparison
            state["final_response"] = summary_text
            state["sources"] = [
                {"filename": comparison["document_a"], "page_number": None},
                {"filename": comparison["document_b"], "page_number": None},
            ]
        except GeminiAPIKeyMissingError as exc:
            state.setdefault("errors", []).append(str(exc))
            state["final_response"] = f"Gemini API key not configured: {exc}"
        except GeminiUnavailableError as exc:
            state.setdefault("errors", []).append(str(exc))
            state["final_response"] = f"Gemini API error: {exc}"
        except Exception as exc:  # noqa: BLE001
            logger.error(f"Multi-agent node failed: {exc}")
            state.setdefault("errors", []).append(f"Multi-agent error: {exc}")
            state["final_response"] = "The multi-agent workflow encountered an error."
        state["execution_trace"] = trace
        return state

    def unsupported_node(state: WorkbenchState) -> WorkbenchState:
        trace = state.get("execution_trace", [])
        trace.append("Orchestrator: request classified as unsupported")
        state["final_response"] = (
            "This request doesn't match summarization, Q&A, or comparison of your indexed documents. "
            "Try rephrasing, or use the Summarize / Ask / Compare tabs directly."
        )
        state["execution_trace"] = trace
        return state

    def route_selector(state: WorkbenchState) -> str:
        agent = state.get("selected_agent", "unsupported")
        return agent if agent in {"summarizer", "qa", "comparator", "multi_agent"} else "unsupported"

    graph = StateGraph(WorkbenchState)
    graph.add_node("orchestrator", orchestrator_node)
    graph.add_node("summarizer", summarizer_node)
    graph.add_node("qa", qa_node)
    graph.add_node("comparator", comparator_node)
    graph.add_node("multi_agent", multi_agent_node)
    graph.add_node("unsupported", unsupported_node)

    graph.set_entry_point("orchestrator")
    graph.add_conditional_edges(
        "orchestrator",
        route_selector,
        {
            "summarizer": "summarizer",
            "qa": "qa",
            "comparator": "comparator",
            "multi_agent": "multi_agent",
            "unsupported": "unsupported",
        },
    )
    graph.add_edge("summarizer", END)
    graph.add_edge("qa", END)
    graph.add_edge("comparator", END)
    graph.add_edge("multi_agent", END)
    graph.add_edge("unsupported", END)

    return graph.compile(), llm_client


def run_workflow(
    user_query: str,
    model: str,
    temperature: float,
    top_k: int,
    selected_documents: Optional[list] = None,
    mode_hint: Optional[str] = None,
    detail_level: str = "concise",
) -> WorkbenchState:
    compiled_graph, llm_client = build_workflow(model=model, temperature=temperature, top_k=top_k)

    initial_state: WorkbenchState = {
        "user_query": user_query,
        "selected_documents": selected_documents or [],
        "mode_hint": mode_hint,
        "retrieved_context": [],
        "intermediate_results": {"detail_level": detail_level},
        "sources": [],
        "errors": [],
        "execution_trace": [],
        "llm_model": model,
    }

    start = time.time()
    final_state = compiled_graph.invoke(initial_state)
    final_state["processing_time_seconds"] = round(time.time() - start, 2)
    return final_state
