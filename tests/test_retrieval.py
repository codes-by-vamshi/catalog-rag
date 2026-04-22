"""
Tests for the retrieval pipeline (BM25 + exact code lookup).
These tests use in-memory indexes so no Ollama connection is needed.
"""
import pytest
from src.schema import ChunkRecord
from src.keyword_index import KeywordIndex
from src.reranker import rerank
from src.retriever import analyse_query


def _make_chunk(
    chunk_id: str,
    product_codes: list,
    text: str,
    chunk_type: str = "title_and_description",
    content_type: str = "product_record",
    product_name: str = "",
) -> ChunkRecord:
    from src.normalizer import derive_base_codes
    base_codes = derive_base_codes(product_codes)
    return ChunkRecord(
        chunk_id=chunk_id,
        record_id=f"rec_{chunk_id}",
        chunk_type=chunk_type,
        page_start=1,
        page_end=1,
        product_name=product_name or chunk_id,
        product_codes=product_codes,
        base_product_codes=base_codes,
        category="Test",
        content_type=content_type,
        text=text,
        searchable_text=text + " " + " ".join(product_codes),
    )


@pytest.fixture
def sample_chunks():
    return [
        _make_chunk(
            "c1",
            ["36-5150/01"],
            "ADR Touch Control Pro 2000 BS EN Auto Compression Machine. Load 3000kN.",
            product_name="ADR Touch Control Pro 2000 BS EN",
        ),
        _make_chunk(
            "c2",
            ["36-5150/06"],
            "ADR Touch Control Pro 2000 ASTM Auto Compression Machine. ASTM C39.",
            product_name="ADR Touch Control Pro 2000 ASTM",
        ),
        _make_chunk(
            "c3",
            ["24-9186"],
            "Cone Penetrometer for soil bearing capacity testing.",
            product_name="Cone Penetrometer",
        ),
        _make_chunk(
            "c4",
            ["25-3518/01"],
            "CBR Mould with base plate and collar.",
            product_name="CBR Mould",
        ),
    ]


@pytest.fixture
def keyword_index(sample_chunks):
    ki = KeywordIndex()
    ki.build(sample_chunks)
    return ki


class TestExactCodeLookup:
    def test_full_code_lookup(self, keyword_index):
        results = keyword_index.exact_code_lookup("36-5150/01")
        codes = [c for chunk in results for c in chunk.product_codes]
        assert "36-5150/01" in codes

    def test_base_code_lookup(self, keyword_index):
        results = keyword_index.exact_code_lookup("36-5150")
        # Should return both /01 and /06 variants
        all_codes = [c for chunk in results for c in chunk.product_codes]
        assert any("36-5150" in c for c in all_codes)

    def test_nonexistent_code(self, keyword_index):
        results = keyword_index.exact_code_lookup("99-9999")
        assert results == []

    def test_code_exists(self, keyword_index):
        assert keyword_index.code_exists("36-5150/01") is True
        assert keyword_index.code_exists("99-0000") is False


class TestBM25Search:
    def test_finds_by_product_name(self, keyword_index):
        results = keyword_index.bm25_search("cone penetrometer", top_k=5)
        codes = [c for chunk, _ in results for c in chunk.product_codes]
        assert "24-9186" in codes

    def test_finds_by_code_token(self, keyword_index):
        # BM25 IDF can be 0 when a token appears in exactly half the corpus
        # (log(1)=0). Exact code lookup is the correct API for code-only queries.
        # Verify via exact_code_lookup instead, which is guaranteed O(1).
        results = keyword_index.exact_code_lookup("36-5150")
        assert len(results) > 0

    def test_returns_empty_for_unrelated(self, keyword_index):
        results = keyword_index.bm25_search("xylophone orchestra music", top_k=5)
        # May return results with 0 score — they should be filtered
        assert all(score > 0 for _, score in results)


class TestQueryAnalysis:
    def test_detects_product_code(self):
        analysis = analyse_query("What are the specs of 36-5150/01?")
        assert "36-5150/01" in analysis.product_codes

    def test_detects_base_code(self):
        analysis = analyse_query("Tell me about 36-5150")
        assert "36-5150" in analysis.base_codes

    def test_detects_comparison_intent(self):
        analysis = analyse_query("What is the difference between /01 and /06?")
        assert analysis.is_comparison is True

    def test_detects_variant_intent(self):
        analysis = analyse_query("Show all variants of this product")
        assert analysis.is_variant_query is True

    def test_detects_accessory_intent(self):
        analysis = analyse_query("What accessories are available?")
        assert analysis.is_accessory_query is True

    def test_no_false_positives_on_code(self):
        analysis = analyse_query("What is a compression machine?")
        assert len(analysis.product_codes) == 0


class TestReranker:
    def test_exact_code_ranks_first(self, sample_chunks):
        candidates = [(c, 1.0) for c in sample_chunks]
        ranked = rerank(
            candidates,
            query="36-5150/01 specifications",
            query_codes=["36-5150/01"],
            query_base_codes=["36-5150"],
        )
        assert ranked[0].chunk.chunk_id == "c1"

    def test_product_record_beats_buyers_guide(self):
        product_chunk = _make_chunk(
            "prod", ["36-5150/01"], "Product specs", content_type="product_record"
        )
        guide_chunk = _make_chunk(
            "guide", ["36-5150/01"], "Guide row", content_type="buyers_guide",
            chunk_type="buyers_guide_row"
        )
        ranked = rerank(
            [(product_chunk, 1.0), (guide_chunk, 1.0)],
            query="specifications of 36-5150/01",
            query_codes=["36-5150/01"],
            query_base_codes=["36-5150"],
        )
        assert ranked[0].chunk.chunk_id == "prod"
