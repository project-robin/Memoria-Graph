from __future__ import annotations

import os
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[2]
LOCAL_DATA_DIR = ROOT_DIR / ".local_data"
TEMP_IMAGE_DIR = ROOT_DIR / "temp_images"
QDRANT_DIR = LOCAL_DATA_DIR / "qdrant"
THUMBNAIL_DIR = LOCAL_DATA_DIR / "thumbnails"
FACE_DIR = LOCAL_DATA_DIR / "faces"
DB_PATH = LOCAL_DATA_DIR / "metadata.sqlite3"
CHECKPOINT_DB_PATH = LOCAL_DATA_DIR / "langgraph_checkpoints.sqlite3"

APP_TITLE = "Canvas Query"
APP_SUBTITLE = (
    "Agentic photo-memory retrieval with LangGraph, persistent indexing, multimodal "
    "embeddings, metadata fusion, and clarification-aware search sessions."
)

EMBEDDING_MODEL = os.environ.get("EMBEDDING_MODEL", "gemini-embedding-2-preview")
QUERY_MODEL = os.environ.get("QUERY_MODEL", "gemini-2.5-flash-lite")
ENRICHMENT_MODEL = os.environ.get("ENRICHMENT_MODEL", QUERY_MODEL)
RERANK_MODEL = os.environ.get("RERANK_MODEL", QUERY_MODEL)
JUDGE_MODEL = os.environ.get("JUDGE_MODEL", "gemma-4-31b-it")

COLLECTION_NAME = "multimodal_images"
VECTOR_SIZE = int(os.environ.get("VECTOR_SIZE", "3072"))
TOP_K_VECTOR = int(os.environ.get("TOP_K_VECTOR", "24"))
TOP_K_METADATA = int(os.environ.get("TOP_K_METADATA", "16"))
TOP_K_RERANK = int(os.environ.get("TOP_K_RERANK", "8"))
TOP_K_JUDGE = int(os.environ.get("TOP_K_JUDGE", "5"))

THUMBNAIL_MAX_EDGE = int(os.environ.get("THUMBNAIL_MAX_EDGE", "1024"))
INDEX_MAX_RETRIES = int(os.environ.get("INDEX_MAX_RETRIES", "3"))
WORKER_BATCH_SIZE = int(os.environ.get("WORKER_BATCH_SIZE", "2"))
WORKER_POLL_SECONDS = float(os.environ.get("WORKER_POLL_SECONDS", "1.0"))

SUPPORTED_MIME_TYPES = {
    "image/jpeg",
    "image/png",
    "image/webp",
}

EMBEDDING_COST_PER_M_TOKENS = 0.15
EMBEDDING_BATCH_COST_PER_M_TOKENS = 0.075
FLASH_LITE_INPUT_COST_PER_M_TOKENS = 0.25
FLASH_LITE_OUTPUT_COST_PER_M_TOKENS = 1.50
ESTIMATED_IMAGE_TOKENS = 258
ESTIMATED_RERANK_PROMPT_TOKENS = 1600
ESTIMATED_JUDGE_OUTPUT_TOKENS = 140


class AppConfigError(RuntimeError):
    """Raised when required runtime configuration is missing."""


def _load_local_env_file() -> None:
    env_path = ROOT_DIR / ".env.local"
    if not env_path.exists():
        return

    for raw_line in env_path.read_text().splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip().strip('"').strip("'")
        if key and value and key not in os.environ:
            os.environ[key] = value


def get_api_key() -> str:
    _load_local_env_file()
    api_key = os.environ.get("GEMINI_API_KEY", "").strip()
    if not api_key:
        raise AppConfigError("GEMINI_API_KEY is required in the environment.")
    return api_key


def ensure_app_dirs() -> None:
    for directory in (
        LOCAL_DATA_DIR,
        TEMP_IMAGE_DIR,
        QDRANT_DIR,
        THUMBNAIL_DIR,
        FACE_DIR,
    ):
        directory.mkdir(parents=True, exist_ok=True)
