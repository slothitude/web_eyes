import os

from dotenv import load_dotenv

load_dotenv()


def _env(key: str, default: str | None = None) -> str:
    val = os.getenv(key, default)
    if val is None:
        raise EnvironmentError(f"Required env var {key} is not set")
    return val


def _env_int(key: str, default: int) -> int:
    return int(os.getenv(key, str(default)))


def _env_float(key: str, default: float) -> float:
    return float(os.getenv(key, str(default)))


# NVIDIA NIM
NIM_API_KEY: str = _env("NIM_API_KEY", "")
NIM_BASE_URL: str = _env("NIM_BASE_URL", "https://integrate.api.nvidia.com/v1")
NIM_MODEL: str = _env("NIM_MODEL", "google/gemma-3-27b-it")

# SearXNG
SEARXNG_HOST: str = _env("SEARXNG_HOST", "localhost")
SEARXNG_PORT: int = _env_int("SEARXNG_PORT", 8888)
SEARXNG_BASE_PATH: str = _env("SEARXNG_BASE_PATH", "/search")

# FastAPI
API_HOST: str = _env("API_HOST", "0.0.0.0")
API_PORT: int = _env_int("API_PORT", 3000)

# MCP server
MCP_HOST: str = _env("MCP_HOST", "0.0.0.0")
MCP_PORT: int = _env_int("MCP_PORT", 3001)

# Crawl settings
DEFAULT_SEARCH_LIMIT: int = _env_int("DEFAULT_SEARCH_LIMIT", 10)
CONTENT_FILTER_THRESHOLD: float = _env_float("CONTENT_FILTER_THRESHOLD", 0.6)
WORD_COUNT_THRESHOLD: int = _env_int("WORD_COUNT_THRESHOLD", 10)

# Vision settings
NIM_VISION_MODEL: str = _env("NIM_VISION_MODEL", "google/gemma-3-27b-it")
VISION_FALLBACK_ENABLED: bool = os.getenv("VISION_FALLBACK_ENABLED", "true").lower() == "true"
VISION_WORD_THRESHOLD: int = _env_int("VISION_WORD_THRESHOLD", 30)
VISION_MAX_IMAGE_DIMENSION: int = _env_int("VISION_MAX_IMAGE_DIMENSION", 1280)
