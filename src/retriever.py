"""
Multi-stage retrieval pipeline.

Stage A: Query analysis (detect codes, intent, standards, etc.)
Stage B: Retrieval (exact code → BM25 → vector)
Stage C: Reranking
Stage D: Disambiguation check
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

from src.config import TOP_K_BM25, TOP_K_VECTOR, TOP_K_FINAL, DISAMBIG_THRESHOLD
from src.schema import ChunkRecord
from src.normalizer import extract_codes, split_code, derive_base_codes, normalise_text
from src.keyword_index import KeywordIndex
from src.vector_index import VectorIndex
from src.reranker import ScoredChunk, rerank
from src.utils import get_logger

log = get_logger(__name__)

_COMPARISON_RE = re.compile(r"\b(differ|difference|compare|vs\.?|versus|between)\b", re.IGNORECASE)
_VARIANT_RE    = re.compile(r"\b(variant|variants|models?|versions?|types?)\b", re.IGNORECASE)
_ACCESSORY_RE  = re.compile(r"\b(accessor\w*|spare\w*|consumable\w*)\b", re.IGNORECASE)
_STANDARDS_RE  = re.compile(r"\b(standard|comply|compliance|BS|EN|ASTM|ISO)\b", re.IGNORECASE)
# Signals that the user expects multiple results (category/discovery intent), not a specific product
_DISCOVERY_RE  = re.compile(
    r"\b(what|which|find|show|where|how|related|equipment|products?|specifications?|pages?|available|looking)\b",
    re.IGNORECASE,
)


@dataclass
class QueryAnalysis:
    raw_query: str
    product_codes: List[str] = field(default_factory=list)
    base_codes: List[str]    = field(default_factory=list)
    is_comparison: bool      = False
    is_variant_query: bool   = False
    is_accessory_query: bool = False
    is_standard_query: bool  = False
    is_discovery_query: bool = False  # user wants category/multiple results, not a specific product


@dataclass
class RetrievalResult:
    query: str
    analysis: QueryAnalysis
    ranked: List[ScoredChunk]
    needs_disambiguation: bool = False
    disambiguation_candidates: List[ScoredChunk] = field(default_factory=list)


def analyse_query(query: str) -> QueryAnalysis:
    codes = extract_codes(query)
    base_codes = derive_base_codes(codes)

    return QueryAnalysis(
        raw_query=query,
        product_codes=codes,
        base_codes=base_codes,
        is_comparison=bool(_COMPARISON_RE.search(query)),
        is_variant_query=bool(_VARIANT_RE.search(query)),
        is_accessory_query=bool(_ACCESSORY_RE.search(query)),
        is_standard_query=bool(_STANDARDS_RE.search(query)),
        is_discovery_query=bool(
            _DISCOVERY_RE.search(query)
            or _STANDARDS_RE.search(query)
            or _ACCESSORY_RE.search(query)
        ),
    )


def retrieve(
    query: str,
    keyword_index: KeywordIndex,
    vector_index: VectorIndex,
) -> RetrievalResult:
    analysis = analyse_query(query)
    log.debug(
        "Query analysis: codes=%s base=%s comparison=%s variant=%s",
        analysis.product_codes,
        analysis.base_codes,
        analysis.is_comparison,
        analysis.is_variant_query,
    )

    candidates: Dict[str, Tuple[ChunkRecord, float]] = {}

    # Stage B-1: Exact code lookup (highest confidence)
    all_codes = analysis.product_codes + analysis.base_codes
    for code in all_codes:
        exact_chunks = keyword_index.exact_code_lookup(code)
        for chunk in exact_chunks:
            # Weight exact code hits very high
            existing_score = candidates.get(chunk.chunk_id, (chunk, 0.0))[1]
            candidates[chunk.chunk_id] = (chunk, max(existing_score, 10.0))

    # Stage B-2: BM25 retrieval
    bm25_results = keyword_index.bm25_search(query, top_k=TOP_K_BM25)
    for chunk, score in bm25_results:
        if chunk.chunk_id in candidates:
            candidates[chunk.chunk_id] = (chunk, candidates[chunk.chunk_id][1] + score)
        else:
            candidates[chunk.chunk_id] = (chunk, score)

    # Stage B-3: Vector retrieval
    try:
        vector_results = vector_index.search(query, top_k=TOP_K_VECTOR)
        for chunk, score in vector_results:
            if chunk.chunk_id in candidates:
                old = candidates[chunk.chunk_id]
                candidates[chunk.chunk_id] = (chunk, old[1] + score * 0.5)
            else:
                candidates[chunk.chunk_id] = (chunk, score * 0.5)
    except Exception as e:
        log.warning("Vector search failed: %s", e)

    all_candidates = list(candidates.values())
    log.debug("Total candidates before reranking: %d", len(all_candidates))

    # Stage C: Reranking
    ranked = rerank(
        all_candidates,
        query=query,
        query_codes=analysis.product_codes,
        query_base_codes=analysis.base_codes,
        top_k=TOP_K_FINAL,
    )

    # Stage D: Disambiguation check
    needs_disambig = False
    disambig_candidates: List[ScoredChunk] = []

    # Discovery/category queries (standards, accessories, topic searches) expect multiple
    # results — never ask for disambiguation; just return the ranked list.
    if (
        len(ranked) >= 2
        and not analysis.product_codes
        and not analysis.is_discovery_query
    ):
        top_score = ranked[0].score
        top_product_codes = set(ranked[0].chunk.product_codes) or set(ranked[0].chunk.base_product_codes)

        # Find the first chunk belonging to a genuinely different product
        alt_chunk: Optional[ScoredChunk] = None
        for sc in ranked[1:]:
            alt_codes = set(sc.chunk.product_codes) or set(sc.chunk.base_product_codes)
            if alt_codes != top_product_codes:
                alt_chunk = sc
                break

        different_products = alt_chunk is not None
        second_score = alt_chunk.score if alt_chunk else top_score

        scores_close = (top_score - second_score) < DISAMBIG_THRESHOLD * max(top_score, 0.1)
        # Short bare-name queries (≤3 words) are inherently ambiguous even if one product dominates
        short_bare_query = len(analysis.raw_query.split()) <= 3

        if different_products and (scores_close or short_bare_query):
            needs_disambig = True
            disambig_candidates = _collect_disambig_candidates(ranked)
            log.info(
                "Disambiguation triggered: top_score=%.3f second_score=%.3f short=%s",
                top_score,
                second_score,
                short_bare_query,
            )

    return RetrievalResult(
        query=query,
        analysis=analysis,
        ranked=ranked,
        needs_disambiguation=needs_disambig,
        disambiguation_candidates=disambig_candidates,
    )


def _collect_disambig_candidates(ranked: List[ScoredChunk]) -> List[ScoredChunk]:
    """Collect up to 5 distinct product candidates for disambiguation."""
    seen_codes: set = set()
    result: List[ScoredChunk] = []
    for sc in ranked:
        key = tuple(sorted(sc.chunk.product_codes)) or (sc.chunk.record_id,)
        if key not in seen_codes:
            seen_codes.add(key)
            result.append(sc)
        if len(result) >= 5:
            break
    return result


def _group_by_base_code(
    ranked: List[ScoredChunk],
    base_codes: List[str],
) -> Dict[str, List[ScoredChunk]]:
    groups: Dict[str, List[ScoredChunk]] = {}
    for sc in ranked:
        for bc in sc.chunk.base_product_codes:
            if bc.upper() in [b.upper() for b in base_codes]:
                groups.setdefault(bc, []).append(sc)
    return groups
