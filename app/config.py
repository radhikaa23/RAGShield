"""Small environment-backed settings shared by the RAG pipeline."""
import os
from pathlib import Path

from dotenv import load_dotenv

PROJECT_ROOT = Path(__file__).resolve().parents[1]
load_dotenv(PROJECT_ROOT / ".env")


def env_int(name: str, default: int, minimum: int = 1) -> int:
    try:
        value = int(os.getenv(name, str(default)))
    except ValueError as exc:
        raise ValueError(f"{name} must be an integer.") from exc
    if value < minimum:
        raise ValueError(f"{name} must be at least {minimum}.")
    return value


def bounded_env_int(name: str, default: int, minimum: int, maximum: int) -> int:
    value = env_int(name, default, minimum)
    if value > maximum:
        raise ValueError(f"{name} must be no greater than {maximum}.")
    return value


def chunk_size() -> int:
    return env_int("RAGSHIELD_CHUNK_SIZE", 500)


def chunk_overlap() -> int:
    return env_int("RAGSHIELD_CHUNK_OVERLAP", 75, minimum=0)


def top_k() -> int:
    return env_int("RAGSHIELD_TOP_K", 5)


def max_pdf_size_mb() -> int:
    return bounded_env_int("RAGSHIELD_MAX_PDF_MB", 20, 1, 100)


def max_pdf_pages() -> int:
    return bounded_env_int("RAGSHIELD_MAX_PDF_PAGES", 300, 1, 1000)


def max_extracted_chars() -> int:
    return bounded_env_int("RAGSHIELD_MAX_EXTRACTED_CHARS", 5_000_000, 1_000, 10_000_000)


def max_distance() -> float:
    try:
        value = float(os.getenv("RAGSHIELD_MAX_DISTANCE", "0.72"))
    except ValueError as exc:
        raise ValueError("RAGSHIELD_MAX_DISTANCE must be a number.") from exc
    if not 0 <= value <= 2:
        raise ValueError("RAGSHIELD_MAX_DISTANCE must be between 0 and 2.")
    return value
