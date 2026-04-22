"""
BM25 keyword index + exact product-code lookup map.

Uses rank_bm25.BM25Okapi for lexical search.
The exact code map (code -> [chunk_ids]) is stored separately as JSON.
"""
from __future__ import annotations

import json
import pickle
import re
from pathlib import Path
from typing import Dict, List, Tuple

from src.config import BM25_PATH, CODE_MAP_PATH
from src.schema import ChunkRecord
from src.normalizer import extract_codes, split_code, normalise_text
from src.utils import get_logger, ensure_dirs

log = get_logger(__name__)

_TOKENISE_RE = re.compile(r"[^\w]+")


def _tokenise(text: str) -> List[str]:
    text = normalise_text(text)
    return [t for t in _TOKENISE_RE.split(text) if len(t) > 1]


class KeywordIndex:
    def __init__(self) -> None:
        self._chunks: List[ChunkRecord] = []
        self._bm25 = None
        self._code_map: Dict[str, List[str]] = {}  # code -> [chunk_id, ...]

    # ── Build ──────────────────────────────────────────────────────────────────

    def build(self, chunks: List[ChunkRecord]) -> None:
        from rank_bm25 import BM25Okapi  # type: ignore

        # Deduplicate by chunk_id
        seen: set = set()
        deduped: List[ChunkRecord] = []
        for c in chunks:
            if c.chunk_id not in seen:
                seen.add(c.chunk_id)
                deduped.append(c)
        chunks = deduped

        self._chunks = chunks
        self._code_map = {}

        corpus: List[List[str]] = []
        for chunk in chunks:
            tokens = _tokenise(chunk.searchable_text)
            corpus.append(tokens)

            # Exact code map
            for code in chunk.product_codes:
                self._code_map.setdefault(code.upper(), []).append(chunk.chunk_id)
                base, _ = split_code(code)
                self._code_map.setdefault(base.upper(), []).append(chunk.chunk_id)

        self._bm25 = BM25Okapi(corpus)
        log.info(
            "BM25 index built: %d chunks, %d code entries",
            len(chunks),
            len(self._code_map),
        )

    # ── Search ─────────────────────────────────────────────────────────────────

    def bm25_search(self, query: str, top_k: int = 15) -> List[Tuple[ChunkRecord, float]]:
        if self._bm25 is None:
            raise RuntimeError("Index not built. Call build() or load() first.")
        tokens = _tokenise(query)
        scores = self._bm25.get_scores(tokens)
        # Pair with chunks, sort descending
        pairs = sorted(
            zip(self._chunks, scores.tolist()),
            key=lambda x: x[1],
            reverse=True,
        )
        return [(chunk, score) for chunk, score in pairs[:top_k] if score > 0]

    def exact_code_lookup(self, code: str) -> List[ChunkRecord]:
        """Return all chunks that contain the given product code (exact match)."""
        code_upper = code.strip().upper()
        ids = set(self._code_map.get(code_upper, []))
        base, _ = split_code(code_upper)
        ids |= set(self._code_map.get(base, []))

        id_to_chunk = {c.chunk_id: c for c in self._chunks}
        return [id_to_chunk[cid] for cid in ids if cid in id_to_chunk]

    def code_exists(self, code: str) -> bool:
        return code.strip().upper() in self._code_map

    # ── Persist ────────────────────────────────────────────────────────────────

    def save(self, bm25_path: Path = BM25_PATH, code_map_path: Path = CODE_MAP_PATH) -> None:
        ensure_dirs(bm25_path.parent)
        with bm25_path.open("wb") as f:
            pickle.dump({"bm25": self._bm25, "chunks": self._chunks}, f)
        with code_map_path.open("w", encoding="utf-8") as f:
            json.dump(self._code_map, f, indent=2)
        log.info("Keyword index saved to %s", bm25_path)

    def load(self, bm25_path: Path = BM25_PATH, code_map_path: Path = CODE_MAP_PATH) -> None:
        if not bm25_path.exists():
            raise FileNotFoundError(f"BM25 index not found at {bm25_path}. Run build-index first.")
        with bm25_path.open("rb") as f:
            data = pickle.load(f)
        self._bm25 = data["bm25"]
        self._chunks = data["chunks"]
        if code_map_path.exists():
            with code_map_path.open("r", encoding="utf-8") as f:
                self._code_map = json.load(f)
        log.info("Keyword index loaded: %d chunks", len(self._chunks))
