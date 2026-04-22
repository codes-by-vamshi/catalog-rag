"""
Minimal Streamlit web UI for the Catalogue RAG system.

Run with:
    streamlit run src/app.py
"""
from __future__ import annotations

import sys
from pathlib import Path

# Allow running from project root
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import streamlit as st

from src.config import BM25_PATH, CODE_MAP_PATH
from src.keyword_index import KeywordIndex
from src.vector_index import VectorIndex
from src.retriever import retrieve, RetrievalResult
from src.answerer import generate_answer, format_disambiguation, format_context


@st.cache_resource(show_spinner="Loading indexes…")
def load_indexes():
    ki = KeywordIndex()
    vi = VectorIndex()
    ki.load(BM25_PATH, CODE_MAP_PATH)
    vi.load()
    return ki, vi


def main():
    st.set_page_config(
        page_title="Product Catalogue RAG",
        page_icon="📚",
        layout="wide",
    )
    st.title("📚 Product Catalogue Assistant")
    st.caption("Local RAG system — answers grounded in your PDF catalogue")

    # ── Sidebar ────────────────────────────────────────────────────────────────
    with st.sidebar:
        st.header("Settings")
        show_evidence = st.checkbox("Show retrieved evidence", value=True)
        show_debug    = st.checkbox("Show retrieval debug info", value=False)
        st.markdown("---")
        st.markdown(
            "**Tips:**\n"
            "- Use product codes like `36-5150/01` for precise results\n"
            "- Ask about specs, standards, accessories\n"
            "- Ask to compare variants"
        )

    # ── Main area ──────────────────────────────────────────────────────────────
    try:
        ki, vi = load_indexes()
    except Exception as e:
        st.error(
            f"Could not load indexes: {e}\n\n"
            "Run `python -m src.cli build-index` first."
        )
        return

    question = st.text_input(
        "Ask a question about the catalogue:",
        placeholder="e.g. What are the specifications of 36-5150/01?",
    )

    if not question:
        return

    with st.spinner("Retrieving and generating answer…"):
        result: RetrievalResult = retrieve(question, ki, vi)

    # ── Disambiguation ─────────────────────────────────────────────────────────
    if result.needs_disambiguation:
        st.warning("Multiple products match your query. Please choose:")
        options = []
        for i, sc in enumerate(result.disambiguation_candidates):
            c = sc.chunk
            label = f"{c.product_name or '(unnamed)'} — {', '.join(c.product_codes) or 'no code'} — page {c.page_start}"
            options.append(label)

        chosen = st.radio("Select a product:", options)
        if chosen:
            chosen_idx = options.index(chosen)
            chosen_code = result.disambiguation_candidates[chosen_idx].chunk.product_codes
            if chosen_code:
                refined_q = f"{question} [{chosen_code[0]}]"
                with st.spinner("Re-retrieving for chosen product…"):
                    result = retrieve(refined_q, ki, vi)
            else:
                st.info("Could not narrow down — showing best match.")

    # ── Answer ─────────────────────────────────────────────────────────────────
    if not result.needs_disambiguation or result.ranked:
        answer = generate_answer(question, result)
        st.markdown("### Answer")
        st.markdown(answer)

    # ── Evidence viewer ────────────────────────────────────────────────────────
    if show_evidence and result.ranked:
        st.markdown("---")
        st.markdown("### Retrieved Evidence")
        for i, sc in enumerate(result.ranked, 1):
            c = sc.chunk
            label = (
                f"[{i}] {c.product_name or 'Unnamed'} "
                f"| Codes: {', '.join(c.product_codes) or 'N/A'} "
                f"| Page {c.page_start} "
                f"| Type: {c.chunk_type} "
                f"| Score: {sc.score:.3f}"
            )
            with st.expander(label):
                st.text(c.text[:1200])

    # ── Debug ──────────────────────────────────────────────────────────────────
    if show_debug:
        st.markdown("---")
        st.markdown("### Debug: Query Analysis")
        st.json({
            "product_codes": result.analysis.product_codes,
            "base_codes": result.analysis.base_codes,
            "is_comparison": result.analysis.is_comparison,
            "is_variant_query": result.analysis.is_variant_query,
            "is_accessory_query": result.analysis.is_accessory_query,
            "needs_disambiguation": result.needs_disambiguation,
        })

        st.markdown("### Debug: Score Breakdown")
        for sc in result.ranked:
            st.json({
                "chunk_id": sc.chunk.chunk_id,
                "score": sc.score,
                "breakdown": sc.score_breakdown,
                "product_codes": sc.chunk.product_codes,
                "chunk_type": sc.chunk.chunk_type,
            })


if __name__ == "__main__":
    main()
