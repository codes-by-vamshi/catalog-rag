"""
Product-aware reranker.

Takes candidate chunks from multiple retrieval stages and assigns a unified
priority score that heavily favours exact product-code matches.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

from src.schema import ChunkRecord
from src.normalizer import extract_codes, split_code, normalise_text
from src.utils import get_logger

log = get_logger(__name__)

# Content-type base score (product_record is most trustworthy for product questions)
_CONTENT_TYPE_SCORE: Dict[str, float] = {
    "product_record": 0.10,
    "buyers_guide":   0.04,
    "category_intro": 0.02,
    "index_page":     0.01,
    "other":          0.00,
}

# Chunk-type preference when answering product questions
_CHUNK_TYPE_SCORE: Dict[str, float] = {
    "title_and_description": 0.08,
    "full_record":           0.06,
    "specifications":        0.05,
    "accessories_and_spares": 0.04,
    "models_and_codes":      0.04,
    "buyers_guide_row":      0.02,
    "other":                 0.00,
}


@dataclass
class ScoredChunk:
    chunk: ChunkRecord
    score: float
    score_breakdown: Dict[str, float] = field(default_factory=dict)


def rerank(
    candidates: List[Tuple[ChunkRecord, float]],
    query: str,
    query_codes: List[str],
    query_base_codes: List[str],
    top_k: int = 6,
) -> List[ScoredChunk]:
    """
    Score and rank candidate (chunk, raw_score) pairs.

    Priority order (additive bonuses):
      1. Exact full product code match            +1.00
      2. Exact base code match                   +0.60
      3. Partial code token match                +0.30
      4. Exact normalised product name match     +0.40
      5. Alias match                             +0.20
      6. BM25/vector score (normalised)          +up to 0.30
      7. Content-type preference                 +up to 0.10
      8. Chunk-type preference                   +up to 0.08
    """
    query_norm = normalise_text(query)
    scored: List[ScoredChunk] = []

    # Normalise raw scores to [0, 1] range
    raw_max = max((s for _, s in candidates), default=1.0) or 1.0

    for chunk, raw_score in candidates:
        breakdown: Dict[str, float] = {}
        score = 0.0

        normalised_raw = min(raw_score / raw_max, 1.0) * 0.30
        score += normalised_raw
        breakdown["raw_score"] = normalised_raw

        chunk_codes = [c.upper() for c in chunk.product_codes]
        chunk_base_codes = [c.upper() for c in chunk.base_product_codes]

        # Exact full code match
        for qc in query_codes:
            if qc.upper() in chunk_codes:
                score += 1.00
                breakdown["exact_full_code"] = 1.00
                break

        # Exact base code match
        for qb in query_base_codes:
            if qb.upper() in chunk_base_codes:
                score += 0.60
                breakdown["exact_base_code"] = 0.60
                break

        # Partial code token presence in text
        for qc in query_codes + query_base_codes:
            if qc.upper() in chunk.text.upper():
                score += 0.30
                breakdown["code_in_text"] = 0.30
                break

        # Normalised product name match
        if chunk.product_name:
            chunk_name_norm = normalise_text(chunk.product_name)
            if chunk_name_norm and chunk_name_norm in query_norm:
                score += 0.40
                breakdown["name_in_query"] = 0.40
            elif query_norm and query_norm in chunk_name_norm:
                score += 0.25
                breakdown["query_in_name"] = 0.25

        # Content-type preference
        ct_bonus = _CONTENT_TYPE_SCORE.get(chunk.content_type, 0.0)
        score += ct_bonus
        breakdown["content_type"] = ct_bonus

        # Chunk-type preference
        ckt_bonus = _CHUNK_TYPE_SCORE.get(chunk.chunk_type, 0.0)
        score += ckt_bonus
        breakdown["chunk_type"] = ckt_bonus

        scored.append(ScoredChunk(chunk=chunk, score=score, score_breakdown=breakdown))

    scored.sort(key=lambda x: x.score, reverse=True)

    # Deduplicate: prefer highest-scoring chunk per record_id, but keep up to 3 per record
    # (3 allows title_and_description + specifications + accessories for a single product)
    seen_records: Dict[str, int] = {}
    deduped: List[ScoredChunk] = []
    for sc in scored:
        rid = sc.chunk.record_id
        seen_records[rid] = seen_records.get(rid, 0) + 1
        if seen_records[rid] <= 3:
            deduped.append(sc)

    return deduped[:top_k]
