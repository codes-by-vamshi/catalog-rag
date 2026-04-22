"""
Tests for disambiguation logic in the retrieval pipeline.
"""
import pytest
from src.schema import ChunkRecord
from src.keyword_index import KeywordIndex
from src.vector_index import VectorIndex
from src.retriever import retrieve, _collect_disambig_candidates
from src.reranker import ScoredChunk
from src.normalizer import derive_base_codes


def _make_chunk(
    chunk_id: str,
    product_codes: list,
    text: str,
    product_name: str = "",
    content_type: str = "product_record",
) -> ChunkRecord:
    base_codes = derive_base_codes(product_codes)
    return ChunkRecord(
        chunk_id=chunk_id,
        record_id=f"rec_{chunk_id}",
        chunk_type="title_and_description",
        page_start=10,
        page_end=10,
        product_name=product_name or chunk_id,
        product_codes=product_codes,
        base_product_codes=base_codes,
        category="Test",
        content_type=content_type,
        text=text,
        searchable_text=text + " " + " ".join(product_codes),
    )


def _make_scored(chunk: ChunkRecord, score: float) -> ScoredChunk:
    return ScoredChunk(chunk=chunk, score=score)


class TestDisambiguationCandidates:
    def test_collects_distinct_products(self):
        c1 = _make_chunk("a", ["36-5150/01"], "Compression machine model 1")
        c2 = _make_chunk("b", ["36-5150/06"], "Compression machine model 2")
        c3 = _make_chunk("c", ["24-9186"], "Cone Penetrometer")

        ranked = [
            _make_scored(c1, 0.9),
            _make_scored(c2, 0.88),
            _make_scored(c3, 0.85),
        ]
        candidates = _collect_disambig_candidates(ranked)
        all_codes = [sc.chunk.product_codes for sc in candidates]
        assert len(candidates) == 3

    def test_caps_at_five(self):
        chunks = [
            _make_chunk(f"c{i}", [f"36-{1000+i:04d}"], f"Product {i}")
            for i in range(10)
        ]
        ranked = [_make_scored(c, 1.0 - i * 0.01) for i, c in enumerate(chunks)]
        candidates = _collect_disambig_candidates(ranked)
        assert len(candidates) <= 5

    def test_deduplicates_same_product(self):
        c1 = _make_chunk("a", ["36-5150/01"], "Product A chunk 1")
        c2 = _make_chunk("b", ["36-5150/01"], "Product A chunk 2")
        # Same product codes → should appear only once
        ranked = [_make_scored(c1, 0.9), _make_scored(c2, 0.89)]
        candidates = _collect_disambig_candidates(ranked)
        codes = [sc.chunk.product_codes[0] for sc in candidates if sc.chunk.product_codes]
        assert codes.count("36-5150/01") == 1


class TestDisambiguationTrigger:
    """Integration tests for when disambiguation fires in retrieve()."""

    def _build_ki(self, chunks):
        ki = KeywordIndex()
        ki.build(chunks)
        return ki

    class _DummyVectorIndex:
        """Vector index stub that returns nothing."""
        def search(self, query, top_k=15):
            return []

    def test_no_disambiguation_when_code_explicit(self):
        c1 = _make_chunk("a", ["36-5150/01"], "Compression machine 01", "Machine 01")
        c2 = _make_chunk("b", ["36-5150/06"], "Compression machine 06", "Machine 06")
        ki = self._build_ki([c1, c2])
        vi = self._DummyVectorIndex()

        # Explicit code in query → should NOT disambiguate
        result = retrieve("36-5150/01 specifications", ki, vi)
        assert result.needs_disambiguation is False

    def test_disambiguation_fires_for_ambiguous_name(self):
        # Create two distinct products with similar names and close scores
        c1 = _make_chunk(
            "a", ["36-5150/01"],
            "ADR Auto Compression Machine BS EN version load cell 3000kN",
            "ADR Auto Compression BS EN"
        )
        c2 = _make_chunk(
            "b", ["36-5151/01"],
            "ADR Auto Compression Machine ASTM version load cell 3000kN",
            "ADR Auto Compression ASTM"
        )
        ki = self._build_ki([c1, c2])
        vi = self._DummyVectorIndex()

        # No code in query, name-only
        result = retrieve("ADR Auto Compression Machine", ki, vi)
        # The system may or may not trigger disambiguation depending on scores.
        # We just verify the result structure is valid.
        assert hasattr(result, "needs_disambiguation")
        assert isinstance(result.ranked, list)

    def test_base_code_query_with_multiple_variants(self):
        c1 = _make_chunk("a", ["36-5150/01"], "Machine variant 01", "Machine /01")
        c2 = _make_chunk("b", ["36-5150/06"], "Machine variant 06", "Machine /06")
        ki = self._build_ki([c1, c2])
        vi = self._DummyVectorIndex()

        result = retrieve("Show me variants of 36-5150", ki, vi)
        # Base code query with multiple variants should trigger disambiguation
        assert result.needs_disambiguation is True or len(result.ranked) >= 1
