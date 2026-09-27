"""
Local-only logging utility.

IMPORTANT: Never log full confidential document contents. Only log
metadata (filenames, chunk counts, agent names, timings, error types).
"""
import logging
import os
from .config import BASE_DIR

LOG_DIR = os.path.join(BASE_DIR, "data")
LOG_FILE = os.path.join(LOG_DIR, "app.log")


def get_logger(name: str) -> logging.Logger:
    logger = logging.getLogger(name)
    if not logger.handlers:
        logger.setLevel(logging.INFO)
        formatter = logging.Formatter(
            "%(asctime)s | %(levelname)s | %(name)s | %(message)s"
        )
        stream_handler = logging.StreamHandler()
        stream_handler.setFormatter(formatter)
        logger.addHandler(stream_handler)
        try:
            file_handler = logging.FileHandler(LOG_FILE)
            file_handler.setFormatter(formatter)
            logger.addHandler(file_handler)
        except OSError:
            pass
    return logger


def safe_snippet(text: str, max_chars: int = 60) -> str:
    """Return a redacted-length snippet suitable for logging without leaking document content."""
    if text is None:
        return ""
    cleaned = text.replace("\n", " ").strip()
    if len(cleaned) <= max_chars:
        return cleaned
    return cleaned[:max_chars] + f"...[{len(cleaned)} chars total]"
