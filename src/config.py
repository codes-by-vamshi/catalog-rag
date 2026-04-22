"""
Central configuration for the catalogue RAG system.
All paths, model names, and tunable parameters live here.
"""
from __future__ import annotations

import os
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

# ── Paths ──────────────────────────────────────────────────────────────────────
BASE_DIR = Path(__file__).resolve().parent.parent

DATA_DIR        = BASE_DIR / "data"
ARTIFACTS_DIR   = BASE_DIR / "artifacts"
INDEX_STORE_DIR = BASE_DIR / "index_store"
PROMPTS_DIR     = BASE_DIR / "prompts"

PDF_PATH              = DATA_DIR / os.getenv("PDF_FILENAME", "product_catalogue.pdf")
RAW_PAGES_PATH        = ARTIFACTS_DIR / "raw_pages.jsonl"
EXTRACTED_RECORDS_PATH = ARTIFACTS_DIR / "extracted_records.jsonl"
CHUNKS_PATH           = ARTIFACTS_DIR / "chunks.jsonl"
DEBUG_MATCHES_PATH    = ARTIFACTS_DIR / "debug_matches.jsonl"

CHROMA_DIR    = INDEX_STORE_DIR / "chroma"
METADATA_DIR  = INDEX_STORE_DIR / "metadata"
CODE_MAP_PATH = METADATA_DIR / "code_map.json"
BM25_PATH     = METADATA_DIR / "bm25_index.pkl"

# ── Ollama ─────────────────────────────────────────────────────────────────────
OLLAMA_BASE_URL    = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
GENERATION_MODEL   = os.getenv("GENERATION_MODEL", "llama3.1:8b")
EMBEDDING_MODEL    = os.getenv("EMBEDDING_MODEL", "nomic-embed-text")

# ── OpenAI fallback ────────────────────────────────────────────────────────────
OPENAI_API_KEY       = os.getenv("OPENAI_API_KEY", "")
OPENAI_MODEL         = os.getenv("OPENAI_MODEL", "gpt-4o-mini")

# ── Retrieval ──────────────────────────────────────────────────────────────────
TOP_K_VECTOR     = int(os.getenv("TOP_K_VECTOR", "15"))
TOP_K_BM25       = int(os.getenv("TOP_K_BM25", "15"))
TOP_K_FINAL      = int(os.getenv("TOP_K_FINAL", "6"))
DISAMBIG_THRESHOLD = float(os.getenv("DISAMBIG_THRESHOLD", "0.15"))  # score spread that triggers clarification

# ── ChromaDB ───────────────────────────────────────────────────────────────────
CHROMA_COLLECTION = os.getenv("CHROMA_COLLECTION", "catalogue_chunks")

# ── Chunking ───────────────────────────────────────────────────────────────────
MAX_CHUNK_CHARS = int(os.getenv("MAX_CHUNK_CHARS", "2000"))

# ── Logging ────────────────────────────────────────────────────────────────────
LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO")
