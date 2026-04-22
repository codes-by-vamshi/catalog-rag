"""
Embedding interface: uses Ollama's embedding endpoint.

Falls back to sentence-transformers if Ollama is unreachable.

The backend selected at index-build time is persisted to index_store/metadata/embedding_info.json
so that queries always use the same model as was used during indexing.
"""
from __future__ import annotations

import json
import time
from pathlib import Path
from typing import List, Optional

import requests

from src.config import OLLAMA_BASE_URL, EMBEDDING_MODEL, METADATA_DIR
from src.utils import get_logger

log = get_logger(__name__)

_FALLBACK_MODEL = "all-MiniLM-L6-v2"
_ST_MODEL = None  # lazy-loaded sentence-transformers model

# Resolved at runtime: "ollama" or "sentence_transformers"
_LOCKED_BACKEND: Optional[str] = None

EMBEDDING_INFO_PATH = METADATA_DIR / "embedding_info.json"


def _get_st_model():
    global _ST_MODEL
    if _ST_MODEL is None:
        from sentence_transformers import SentenceTransformer  # type: ignore
        log.info("Loading sentence-transformers model: %s", _FALLBACK_MODEL)
        _ST_MODEL = SentenceTransformer(_FALLBACK_MODEL)
    return _ST_MODEL


def lock_backend(backend: str) -> None:
    """Force all subsequent embed calls to use 'ollama' or 'sentence_transformers'."""
    global _LOCKED_BACKEND
    _LOCKED_BACKEND = backend
    log.info("Embedding backend locked to: %s", backend)


def save_embedding_info(backend: str) -> None:
    """Persist the backend used so future queries can match it."""
    METADATA_DIR.mkdir(parents=True, exist_ok=True)
    info = {"backend": backend}
    EMBEDDING_INFO_PATH.write_text(json.dumps(info, indent=2))
    log.info("Embedding info saved: backend=%s", backend)


def load_and_lock_embedding_info() -> Optional[str]:
    """
    Read persisted embedding info and lock the backend.
    Returns the backend name, or None if no info file exists.
    """
    if not EMBEDDING_INFO_PATH.exists():
        return None
    info = json.loads(EMBEDDING_INFO_PATH.read_text())
    backend = info.get("backend")
    if backend:
        lock_backend(backend)
    return backend


def embed_texts(texts: List[str], batch_size: int = 32) -> List[List[float]]:
    """Return a list of embedding vectors, one per input text."""
    if _LOCKED_BACKEND == "sentence_transformers":
        return _embed_with_st(texts)
    if _LOCKED_BACKEND == "ollama":
        return _embed_with_ollama(texts, batch_size)

    # Auto-detect: try Ollama first, fall back to sentence-transformers
    try:
        result = _embed_with_ollama(texts, batch_size)
        # Don't lock here — lock is set by build()/load() explicitly
        return result
    except Exception as e:
        log.warning("Ollama embedding failed (%s), switching to sentence-transformers", e)
        return _embed_with_st(texts)


def embed_query(query: str) -> List[float]:
    return embed_texts([query])[0]


def detect_backend() -> str:
    """
    Determine which backend is currently usable.
    Returns 'ollama' or 'sentence_transformers'.
    """
    try:
        resp = requests.post(
            f"{OLLAMA_BASE_URL}/api/embed",
            json={"model": EMBEDDING_MODEL, "input": ["ping"]},
            timeout=5,
        )
        if resp.status_code == 200:
            return "ollama"
    except Exception:
        pass
    return "sentence_transformers"


def _embed_with_ollama(texts: List[str], batch_size: int) -> List[List[float]]:
    url = f"{OLLAMA_BASE_URL}/api/embed"
    all_embeddings: List[List[float]] = []

    for i in range(0, len(texts), batch_size):
        batch = texts[i:i + batch_size]
        payload = {"model": EMBEDDING_MODEL, "input": batch}

        resp = requests.post(url, json=payload, timeout=120)
        if resp.status_code != 200:
            raise RuntimeError(f"Ollama embed returned {resp.status_code}: {resp.text[:200]}")

        data = resp.json()
        embeddings = data.get("embeddings") or data.get("embedding")
        if embeddings is None:
            raise RuntimeError(f"Unexpected Ollama embed response: {list(data.keys())}")

        if isinstance(embeddings[0], float):
            embeddings = [embeddings]

        all_embeddings.extend(embeddings)
        if i + batch_size < len(texts):
            time.sleep(0.05)

    return all_embeddings


def _embed_with_st(texts: List[str]) -> List[List[float]]:
    model = _get_st_model()
    vectors = model.encode(texts, show_progress_bar=False, convert_to_numpy=True)
    return [v.tolist() for v in vectors]
