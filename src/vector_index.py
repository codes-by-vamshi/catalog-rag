"""
Vector index backed by ChromaDB (local persistent).
"""
from __future__ import annotations

from pathlib import Path
from typing import List, Tuple

from src.config import CHROMA_DIR, CHROMA_COLLECTION
from src.schema import ChunkRecord
from src.embeddings import (
    embed_texts, embed_query, detect_backend,
    lock_backend, save_embedding_info, load_and_lock_embedding_info,
)
from src.utils import get_logger

log = get_logger(__name__)

_BATCH_SIZE = 64  # chunks to embed and upsert at once


class VectorIndex:
    def __init__(self) -> None:
        self._client = None
        self._collection = None

    def _get_client(self):
        if self._client is None:
            import chromadb  # type: ignore
            self._client = chromadb.PersistentClient(path=str(CHROMA_DIR))
        return self._client

    def _get_collection(self, create: bool = False):
        client = self._get_client()
        if create:
            # Delete existing collection if rebuilding
            try:
                client.delete_collection(CHROMA_COLLECTION)
            except Exception:
                pass
            self._collection = client.create_collection(
                name=CHROMA_COLLECTION,
                metadata={"hnsw:space": "cosine"},
            )
        else:
            self._collection = client.get_collection(CHROMA_COLLECTION)
        return self._collection

    # ── Build ──────────────────────────────────────────────────────────────────

    def build(self, chunks: List[ChunkRecord]) -> None:
        collection = self._get_collection(create=True)

        # Deduplicate by chunk_id (content-addressed IDs guarantee same ID = same content)
        seen: set = set()
        deduped: List[ChunkRecord] = []
        for c in chunks:
            if c.chunk_id not in seen:
                seen.add(c.chunk_id)
                deduped.append(c)
        if len(deduped) < len(chunks):
            log.info("Deduplicated %d → %d chunks", len(chunks), len(deduped))
        chunks = deduped

        # Detect and lock the backend BEFORE embedding so it stays consistent
        backend = detect_backend()
        lock_backend(backend)
        save_embedding_info(backend)
        log.info("Using embedding backend: %s", backend)

        total = len(chunks)
        log.info("Embedding and indexing %d chunks (batch_size=%d)…", total, _BATCH_SIZE)

        for start in range(0, total, _BATCH_SIZE):
            batch = chunks[start:start + _BATCH_SIZE]
            texts = [c.searchable_text for c in batch]

            embeddings = embed_texts(texts)

            ids = [c.chunk_id for c in batch]
            documents = [c.text for c in batch]
            metadatas = [_chunk_meta(c) for c in batch]

            collection.upsert(
                ids=ids,
                embeddings=embeddings,
                documents=documents,
                metadatas=metadatas,
            )
            log.debug("Upserted batch %d/%d", min(start + _BATCH_SIZE, total), total)

        log.info("Vector index built: %d vectors in collection '%s'", total, CHROMA_COLLECTION)

    # ── Search ─────────────────────────────────────────────────────────────────

    def search(self, query: str, top_k: int = 15) -> List[Tuple[ChunkRecord, float]]:
        if self._collection is None:
            self._get_collection(create=False)

        query_vec = embed_query(query)
        results = self._collection.query(
            query_embeddings=[query_vec],
            n_results=top_k,
            include=["documents", "metadatas", "distances"],
        )

        chunks_and_scores: List[Tuple[ChunkRecord, float]] = []
        if not results["ids"] or not results["ids"][0]:
            return chunks_and_scores

        for doc, meta, dist in zip(
            results["documents"][0],
            results["metadatas"][0],
            results["distances"][0],
        ):
            # Cosine distance → similarity
            similarity = max(0.0, 1.0 - dist)
            chunk = _meta_to_chunk(meta, doc)
            chunks_and_scores.append((chunk, similarity))

        return chunks_and_scores

    def load(self) -> None:
        try:
            # Restore the embedding backend that was used at build time
            backend = load_and_lock_embedding_info()
            if backend:
                log.info("Embedding backend restored from index: %s", backend)
            else:
                log.warning(
                    "No embedding_info.json found — backend may mismatch. "
                    "Run build-index to regenerate."
                )
            self._get_collection(create=False)
            log.info("Vector index loaded from %s", CHROMA_DIR)
        except Exception as e:
            raise RuntimeError(
                f"Vector index not found at {CHROMA_DIR}. Run build-index first. ({e})"
            )


def _chunk_meta(chunk: ChunkRecord) -> dict:
    return {
        "chunk_id": chunk.chunk_id,
        "record_id": chunk.record_id,
        "chunk_type": chunk.chunk_type,
        "page_start": chunk.page_start,
        "page_end": chunk.page_end,
        "product_name": chunk.product_name or "",
        "product_codes": ",".join(chunk.product_codes),
        "base_product_codes": ",".join(chunk.base_product_codes),
        "category": chunk.category or "",
        "content_type": chunk.content_type,
    }


def _meta_to_chunk(meta: dict, doc: str) -> ChunkRecord:
    return ChunkRecord(
        chunk_id=meta["chunk_id"],
        record_id=meta["record_id"],
        chunk_type=meta["chunk_type"],
        page_start=int(meta["page_start"]),
        page_end=int(meta["page_end"]),
        product_name=meta["product_name"] or None,
        product_codes=[c for c in meta["product_codes"].split(",") if c],
        base_product_codes=[c for c in meta["base_product_codes"].split(",") if c],
        category=meta["category"] or None,
        content_type=meta["content_type"],
        text=doc,
        searchable_text=doc,
    )
