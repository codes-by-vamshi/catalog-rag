"""
Record-aware chunker.

Rather than naive sliding-window over raw text, we create typed chunks
from each ProductRecord so that retrieval can target specific sub-sections.
"""
from __future__ import annotations

import hashlib
from typing import List

from src.config import CHUNKS_PATH, MAX_CHUNK_CHARS
from src.schema import ProductRecord, ChunkRecord
from src.utils import get_logger, write_jsonl, ensure_dirs

log = get_logger(__name__)


def _chunk_id(record_id: str, chunk_type: str, text: str, index: int = 0) -> str:
    # Content-addressed: include text hash so no two chunks with different content collide
    key = f"{record_id}|{chunk_type}|{index}|{text[:200]}"
    return "chk_" + hashlib.md5(key.encode()).hexdigest()[:16]


def _trim(text: str, max_chars: int = MAX_CHUNK_CHARS) -> str:
    if len(text) <= max_chars:
        return text
    # Try to trim at a sentence boundary
    cutoff = text.rfind(". ", 0, max_chars)
    if cutoff == -1:
        cutoff = max_chars
    return text[:cutoff + 1].strip()


def _base_meta(record: ProductRecord) -> dict:
    return dict(
        record_id=record.record_id,
        page_start=record.page_start,
        page_end=record.page_end,
        product_name=record.product_name,
        product_codes=record.product_codes,
        base_product_codes=record.base_product_codes,
        category=record.category,
        content_type=record.content_type,
    )


def chunk_record(record: ProductRecord) -> List[ChunkRecord]:
    chunks: List[ChunkRecord] = []
    meta = _base_meta(record)

    def add(chunk_type: str, text: str, index: int = 0) -> None:
        text = text.strip()
        if not text:
            return
        searchable = " ".join(filter(None, [
            record.product_name,
            " ".join(record.product_codes),
            " ".join(record.aliases[:5]),
            text,
        ]))
        chunks.append(ChunkRecord(
            chunk_id=_chunk_id(record.record_id, chunk_type, text, index),
            chunk_type=chunk_type,
            text=_trim(text),
            searchable_text=_trim(searchable),
            **meta,
        ))

    ct = record.content_type

    if ct == "product_record":
        # 1. Title + description chunk
        title_desc_parts = []
        if record.product_name:
            title_desc_parts.append(record.product_name)
        if record.product_codes:
            title_desc_parts.append("Product Codes: " + ", ".join(record.product_codes))
        if record.standards:
            title_desc_parts.append("Standards: " + "; ".join(record.standards))
        if record.description:
            title_desc_parts.append(record.description)
        add("title_and_description", "\n".join(title_desc_parts))

        # 2. Specifications chunk
        if record.specifications_text:
            add("specifications", record.specifications_text)

        # 3. Accessories + spares chunk
        acc_parts = []
        if record.accessories_text:
            acc_parts.append("Accessories:\n" + record.accessories_text)
        if record.spares_text:
            acc_parts.append("Spares/Consumables:\n" + record.spares_text)
        if acc_parts:
            add("accessories_and_spares", "\n\n".join(acc_parts))

        # 4. Models / ordering information
        if record.models_text:
            add("models_and_codes", record.models_text)

        # 5. Full record (for cases where we need everything together)
        full = record.raw_text
        if len(full) > 200:
            add("full_record", full)

    elif ct == "buyers_guide":
        # Split buyers guide into rows for better granularity
        lines = [ln.strip() for ln in record.raw_text.splitlines() if ln.strip()]
        # Group into chunks of ~8 lines
        group_size = 8
        for i in range(0, len(lines), group_size):
            group = "\n".join(lines[i:i + group_size])
            add("buyers_guide_row", group, index=i // group_size)

    else:
        # category_intro, index_page, other → one chunk with the raw text
        add("other", record.raw_text)

    return chunks


def chunk_records(records: List[ProductRecord]) -> List[ChunkRecord]:
    all_chunks: List[ChunkRecord] = []
    for record in records:
        all_chunks.extend(chunk_record(record))
    log.info("Created %d chunks from %d records", len(all_chunks), len(records))
    return all_chunks


def save_chunks(chunks: List[ChunkRecord], out_path=CHUNKS_PATH) -> None:
    ensure_dirs(out_path.parent)
    count = write_jsonl(out_path, chunks)
    log.info("Saved %d chunks to %s", count, out_path)


def run(records: List[ProductRecord]) -> List[ChunkRecord]:
    chunks = chunk_records(records)
    save_chunks(chunks)
    return chunks
