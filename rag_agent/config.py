from __future__ import annotations

import os
from pathlib import Path


# Base directories
BASE_DIR: Path = Path(__file__).resolve().parent.parent
DATA_DIR: Path = BASE_DIR / "data"
UPLOADS_DIR: Path = DATA_DIR / "uploads"
INDEX_PATH: Path = DATA_DIR / "index.faiss"
METADATA_PATH: Path = DATA_DIR / "metadata.jsonl"


# Embeddings
EMBEDDING_MODEL_NAME: str = "sentence-transformers/all-MiniLM-L6-v2"


# Retrieval
TOP_K: int = 5


# Chunking
CHUNK_SIZE: int = 750  # characters
CHUNK_OVERLAP: int = 150  # characters
MAX_CHUNKS_PER_PAGE: int = 50


def ensure_directories() -> None:
	"""Create required directories if they don't exist."""
	DATA_DIR.mkdir(parents=True, exist_ok=True)
	UPLOADS_DIR.mkdir(parents=True, exist_ok=True)


def get_openai_api_key() -> str | None:
	"""Read OpenAI API key from environment, if present."""
	return os.environ.get("OPENAI_API_KEY")

