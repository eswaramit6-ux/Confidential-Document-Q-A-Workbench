"""
Shared LangGraph state object. Agents communicate exclusively through
this typed state as it flows through the workflow graph.
"""
from typing import Any, Dict, List, Optional, TypedDict


class WorkbenchState(TypedDict, total=False):
    # Input
    user_query: str
    selected_documents: List[str]  # list of sha256 hashes, when explicitly chosen by the user
    mode_hint: Optional[str]  # UI can force a route, e.g. "summarizer", "comparator"

    # Orchestrator output
    selected_agent: str          # summarizer | qa | comparator | multi_agent | unsupported
    routing_reason: str

    # Retrieval / intermediate results
    retrieved_context: List[Dict[str, Any]]
    intermediate_results: Dict[str, Any]

    # Final output
    final_response: str
    sources: List[Dict[str, Any]]
    errors: List[str]

    # Explainability / metadata
    execution_trace: List[str]
    llm_model: str
    processing_time_seconds: float
