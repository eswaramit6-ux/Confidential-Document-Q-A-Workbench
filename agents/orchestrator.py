"""
Orchestrator: uses the Gemini LLM to classify user intent into one of the
supported routes. This is NOT a hardcoded if/else keyword router — the
routing decision is made by prompting the LLM for a structured
classification, with safe validation/fallback so that any malformed model
output (or a temporary Gemini API failure) cannot crash the app.
"""
import json
import re
from dataclasses import dataclass
from typing import Optional

from models.gemini_client import GeminiClient
from utils.logger import get_logger

logger = get_logger(__name__)

VALID_ROUTES = {"summarizer", "qa", "comparator", "multi_agent", "unsupported"}

ROUTING_PROMPT = """You are the routing controller for a multi-agent document workbench with these specialist agents:

- "summarizer": produces summaries, key points, findings, conclusions for ONE document.
- "qa": answers a specific question using retrieval-augmented generation over indexed documents.
- "comparator": compares TWO documents/versions and reports differences.
- "multi_agent": the request genuinely needs more than one specialist (e.g. "compare both and summarize the changes").
- "unsupported": the request is unrelated to document summarization, Q&A, or comparison.

Classify the following user request and respond with STRICT JSON ONLY, no markdown fences, no extra text,
in exactly this shape:
{{"route": "<one of summarizer|qa|comparator|multi_agent|unsupported>", "reason": "<one short sentence>"}}

USER REQUEST:
"{query}"

JSON:"""


@dataclass
class RoutingDecision:
    route: str
    reason: str
    raw_model_output: str = ""
    used_fallback: bool = False


class Orchestrator:
    def __init__(self, llm_client: Optional[GeminiClient] = None):
        self.llm = llm_client or GeminiClient()

    def route(self, user_query: str, mode_hint: Optional[str] = None) -> RoutingDecision:
        # A UI-selected mode (e.g. user explicitly clicked "Summarize" tab) can be honored directly,
        # but the core requirement — LLM-based classification — is still exercised for the default
        # "Ask Documents" chat entry point.
        if mode_hint in VALID_ROUTES:
            return RoutingDecision(route=mode_hint, reason="Explicitly selected in the UI.", used_fallback=False)

        if not user_query or not user_query.strip():
            return RoutingDecision(route="unsupported", reason="Empty request.", used_fallback=True)

        prompt = ROUTING_PROMPT.format(query=user_query.strip().replace('"', "'"))

        try:
            raw = self.llm.generate(prompt, temperature=0.0)
        except Exception as exc:  # noqa: BLE001
            logger.error(f"Routing LLM call failed, falling back to keyword heuristic: {exc}")
            return self._fallback_route(user_query, raw_output=str(exc))

        decision = self._parse_routing_output(raw, user_query)
        return decision

    def _parse_routing_output(self, raw: str, user_query: str) -> RoutingDecision:
        candidate = raw.strip()
        # Strip markdown fences if the model added them despite instructions.
        candidate = re.sub(r"^```(json)?|```$", "", candidate, flags=re.MULTILINE).strip()

        # Try to locate a JSON object even if the model added extra text.
        match = re.search(r"\{.*\}", candidate, flags=re.DOTALL)
        json_str = match.group(0) if match else candidate

        try:
            parsed = json.loads(json_str)
            route = str(parsed.get("route", "")).strip().lower()
            reason = str(parsed.get("reason", "")).strip() or "Classified by Gemini."
            if route not in VALID_ROUTES:
                raise ValueError(f"Model returned invalid route: {route!r}")
            return RoutingDecision(route=route, reason=reason, raw_model_output=raw, used_fallback=False)
        except (json.JSONDecodeError, ValueError, AttributeError) as exc:
            logger.warning(f"Could not parse routing JSON ({exc}); using fallback heuristic. Raw: {raw[:200]}")
            return self._fallback_route(user_query, raw_output=raw)

    def _fallback_route(self, user_query: str, raw_output: str = "") -> RoutingDecision:
        """
        Safe fallback ONLY used when the Gemini API is unreachable/misconfigured or returns
        unparsable output — keeps the app from crashing. This is not the primary routing mechanism.
        """
        q = user_query.lower()
        if any(w in q for w in ["compare", "difference", "changed", "vs", "versus"]):
            route = "comparator"
        elif any(w in q for w in ["summarize", "summary", "key points", "overview"]):
            route = "summarizer"
        elif q.strip():
            route = "qa"
        else:
            route = "unsupported"
        return RoutingDecision(
            route=route,
            reason="Fallback heuristic used because the Gemini output could not be parsed or the Gemini API was unavailable.",
            raw_model_output=raw_output,
            used_fallback=True,
        )
