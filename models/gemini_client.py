"""
Thin wrapper around Google's Gemini API via langchain-google-genai. This is
now the ONLY LLM inference path in the application. There is intentionally
no other cloud LLM provider (no OpenAI, no Claude, no Groq) and no local
LLM fallback of any kind.

NOTE: this requires network access to Google's Generative Language API and
a valid GEMINI_API_KEY. Local components (document storage, embeddings,
ChromaDB) remain fully on-disk; only LLM requests leave the machine.
"""
import os
from typing import Optional

from dotenv import load_dotenv

from utils.logger import get_logger

logger = get_logger(__name__)

# Load variables from a local .env file (if present) into the process environment.
# This never overwrites variables already set in the real environment.
load_dotenv()

DEFAULT_GEMINI_MODEL = "gemini-2.5-flash"


class GeminiAPIKeyMissingError(RuntimeError):
    """Raised when GEMINI_API_KEY is not configured."""
    pass


class GeminiUnavailableError(RuntimeError):
    """Raised when a Gemini API call fails for any reason (auth, network, quota, etc.)."""
    pass


def get_api_key() -> Optional[str]:
    """Read the Gemini API key from the environment (populated from .env by load_dotenv above).
    Never hardcode the key, never log it, never return it to the UI."""
    return os.environ.get("GEMINI_API_KEY")


def api_key_configured() -> bool:
    key = get_api_key()
    return bool(key and key.strip() and key.strip() != "your_api_key_here")


class GeminiClient:
    """LangChain-compatible wrapper around ChatGoogleGenerativeAI, with the same
    surface area (`generate`, `model` attribute) as the rest of the codebase
    (agents, orchestrator, workflow) expects."""

    def __init__(self, model: str = DEFAULT_GEMINI_MODEL, temperature: float = 0.1,
                 api_key: Optional[str] = None):
        self.model = model
        self.temperature = temperature
        self._api_key = api_key or get_api_key()
        self._chat_model = None  # lazily constructed so a missing key doesn't crash imports

    def _get_chat_model(self, temperature: Optional[float] = None):
        if not self._api_key or not self._api_key.strip() or self._api_key.strip() == "your_api_key_here":
            raise GeminiAPIKeyMissingError(
                "GEMINI_API_KEY is not set. Create a .env file (see .env.example) with "
                "GEMINI_API_KEY=<your key>, or export it as an environment variable, then restart the app."
            )
        from langchain_google_genai import ChatGoogleGenerativeAI
        effective_temperature = temperature if temperature is not None else self.temperature
        # Rebuild only if temperature or model changed since last call (cheap either way).
        return ChatGoogleGenerativeAI(
            model=self.model,
            google_api_key=self._api_key,
            temperature=effective_temperature,
        )

    def is_configured(self) -> bool:
        return api_key_configured()

    def generate(self, prompt: str, system: Optional[str] = None, temperature: Optional[float] = None) -> str:
        """Single-turn generation. Raises GeminiAPIKeyMissingError if no key is configured,
        or GeminiUnavailableError on any API/connectivity failure."""
        chat_model = self._get_chat_model(temperature=temperature)
        try:
            from langchain_core.messages import HumanMessage, SystemMessage
            messages = []
            if system:
                messages.append(SystemMessage(content=system))
            messages.append(HumanMessage(content=prompt))
            response = chat_model.invoke(messages)
            content = getattr(response, "content", None)
            if content is None:
                content = str(response)
            return content.strip()
        except GeminiAPIKeyMissingError:
            raise
        except Exception as exc:  # noqa: BLE001
            logger.error(f"Gemini generation failed: {type(exc).__name__}")
            raise GeminiUnavailableError(
                f"Could not get a response from Gemini model '{self.model}'. "
                f"Check your GEMINI_API_KEY, network connectivity, and model availability. "
                f"(underlying error type: {type(exc).__name__})"
            ) from exc


def health_check(model: str = DEFAULT_GEMINI_MODEL) -> dict:
    """Lightweight health check: verifies an API key is configured and, if so,
    attempts a minimal round-trip call to confirm connectivity/auth are working."""
    configured = api_key_configured()
    reachable = False
    error_message = ""
    if configured:
        try:
            client = GeminiClient(model=model)
            client.generate("Respond with the single word: OK", temperature=0.0)
            reachable = True
        except GeminiAPIKeyMissingError as exc:
            error_message = str(exc)
        except GeminiUnavailableError as exc:
            error_message = str(exc)
        except Exception as exc:  # noqa: BLE001
            error_message = f"Unexpected error during Gemini health check: {type(exc).__name__}"
    else:
        error_message = "GEMINI_API_KEY is not configured."
    return {
        "api_key_configured": configured,
        "gemini_reachable": reachable,
        "model": model,
        "error": error_message,
    }
