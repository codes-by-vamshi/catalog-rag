"""
Command-line interface for the Catalogue RAG system.

Usage:
    python -m src.cli build-index [--pdf PATH]
    python -m src.cli ask "question"
    python -m src.cli inspect-record CODE
    python -m src.cli debug-retrieval "query"
    python -m src.cli show-chunks CODE
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from src.config import (
    PDF_PATH, RAW_PAGES_PATH, EXTRACTED_RECORDS_PATH, CHUNKS_PATH,
    BM25_PATH, CODE_MAP_PATH,
)
from src.utils import get_logger, read_jsonl

log = get_logger("cli")


# ── Lazy-load indexes so CLI imports stay fast ─────────────────────────────────

def _load_indexes():
    from src.keyword_index import KeywordIndex
    from src.vector_index import VectorIndex

    ki = KeywordIndex()
    vi = VectorIndex()

    ki.load(BM25_PATH, CODE_MAP_PATH)
    vi.load()

    return ki, vi


# ── Subcommands ────────────────────────────────────────────────────────────────

def cmd_build_index(args: argparse.Namespace) -> None:
    from src import pdf_extract, record_parser, chunker
    from src.keyword_index import KeywordIndex
    from src.vector_index import VectorIndex

    pdf_path = Path(args.pdf) if args.pdf else PDF_PATH

    print(f"[1/5] Extracting pages from {pdf_path} …")
    pages = pdf_extract.run(pdf_path)
    print(f"      {len(pages)} pages extracted → {RAW_PAGES_PATH}")

    print("[2/5] Parsing product records …")
    records = record_parser.run(pages)
    print(f"      {len(records)} records → {EXTRACTED_RECORDS_PATH}")

    print("[3/5] Chunking records …")
    chunks = chunker.run(records)
    print(f"      {len(chunks)} chunks → {CHUNKS_PATH}")

    print("[4/5] Building keyword (BM25) index …")
    ki = KeywordIndex()
    ki.build(chunks)
    ki.save(BM25_PATH, CODE_MAP_PATH)
    print(f"      Saved to {BM25_PATH}")

    print("[5/5] Building vector (ChromaDB) index …")
    vi = VectorIndex()
    vi.build(chunks)
    print("      Vector index built.")

    print("\nIndex build complete.")


def cmd_ask(args: argparse.Namespace) -> None:
    from src.retriever import retrieve
    from src.answerer import generate_answer

    question = args.question
    print(f"\nQuestion: {question}\n")

    ki, vi = _load_indexes()

    result = retrieve(question, ki, vi)
    answer = generate_answer(question, result)
    print(answer)


def cmd_inspect_record(args: argparse.Namespace) -> None:
    ki, _ = _load_indexes()

    code = args.code.strip()
    chunks = ki.exact_code_lookup(code)

    if not chunks:
        print(f"No records found for code: {code}")
        sys.exit(1)

    # De-duplicate by record_id
    seen = set()
    for chunk in chunks:
        if chunk.record_id in seen:
            continue
        seen.add(chunk.record_id)
        print(f"\n{'='*60}")
        print(f"Record ID  : {chunk.record_id}")
        print(f"Product    : {chunk.product_name or '(unnamed)'}")
        print(f"Codes      : {', '.join(chunk.product_codes)}")
        print(f"Base codes : {', '.join(chunk.base_product_codes)}")
        print(f"Pages      : {chunk.page_start}–{chunk.page_end}")
        print(f"Type       : {chunk.content_type}")
        print(f"\n--- Text preview ---")
        print(chunk.text[:800])


def cmd_debug_retrieval(args: argparse.Namespace) -> None:
    from src.retriever import retrieve

    query = args.query
    ki, vi = _load_indexes()
    result = retrieve(query, ki, vi)

    print(f"\nQuery      : {query}")
    print(f"Codes found: {result.analysis.product_codes}")
    print(f"Base codes : {result.analysis.base_codes}")
    print(f"Needs disamb: {result.needs_disambiguation}")
    print(f"\nTop {len(result.ranked)} ranked results:\n")

    for i, sc in enumerate(result.ranked, 1):
        c = sc.chunk
        print(f"  [{i}] score={sc.score:.4f}  chunk_type={c.chunk_type}  content={c.content_type}")
        print(f"       name={c.product_name or '(none)'}  codes={c.product_codes}  page={c.page_start}")
        print(f"       breakdown={sc.score_breakdown}")
        print(f"       preview: {c.text[:120].replace(chr(10), ' ')}")
        print()


def cmd_show_chunks(args: argparse.Namespace) -> None:
    ki, _ = _load_indexes()

    code = args.code.strip()
    chunks = ki.exact_code_lookup(code)

    if not chunks:
        print(f"No chunks found for code: {code}")
        sys.exit(1)

    for i, chunk in enumerate(chunks, 1):
        print(f"\n--- Chunk {i} [{chunk.chunk_type}] ---")
        print(chunk.text[:600])


# ── Entry point ────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        prog="python -m src.cli",
        description="Catalogue RAG CLI",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    # build-index
    p_build = sub.add_parser("build-index", help="Extract, parse, and index the PDF catalogue")
    p_build.add_argument("--pdf", default=None, help="Path to PDF (default: data/product_catalogue.pdf)")
    p_build.set_defaults(func=cmd_build_index)

    # ask
    p_ask = sub.add_parser("ask", help="Ask a question about the catalogue")
    p_ask.add_argument("question", help="Your question")
    p_ask.set_defaults(func=cmd_ask)

    # inspect-record
    p_inspect = sub.add_parser("inspect-record", help="Inspect all chunks for a product code")
    p_inspect.add_argument("code", help="Product code (e.g. 36-5150 or 36-5150/01)")
    p_inspect.set_defaults(func=cmd_inspect_record)

    # debug-retrieval
    p_debug = sub.add_parser("debug-retrieval", help="Show retrieval scores for a query")
    p_debug.add_argument("query", help="Query string")
    p_debug.set_defaults(func=cmd_debug_retrieval)

    # show-chunks
    p_chunks = sub.add_parser("show-chunks", help="Show raw chunks for a product code")
    p_chunks.add_argument("code", help="Product code")
    p_chunks.set_defaults(func=cmd_show_chunks)

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
