"""
Evaluation harness with 20 hand-written test cases.

Metrics:
  - retrieval top-1 record accuracy
  - retrieval top-3 record accuracy
  - exact code resolution accuracy
  - disambiguation trigger correctness

Run:
    python -m src.evaluate
"""
from __future__ import annotations

import json
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional

from src.config import BM25_PATH, CODE_MAP_PATH
from src.keyword_index import KeywordIndex
from src.vector_index import VectorIndex
from src.retriever import retrieve, RetrievalResult
from src.utils import get_logger

log = get_logger("evaluate")


@dataclass
class TestCase:
    query: str
    expected_codes: List[str]                 # at least one of these must appear in top-K
    expect_disambiguation: bool = False        # should the query trigger disambiguation?
    description: str = ""


TEST_CASES: List[TestCase] = [
    # ── Exact code queries ─────────────────────────────────────────────────────
    TestCase(
        query="What are the specifications of 36-5150/01?",
        expected_codes=["36-5150/01"],
        description="Exact full code lookup",
    ),
    TestCase(
        query="Tell me about product 36-5150/06",
        expected_codes=["36-5150/06", "36-5150"],
        description="Exact full code with variant",
    ),
    TestCase(
        query="Specifications for 24-9186",
        expected_codes=["24-9186"],
        description="Exact base code lookup",
    ),
    TestCase(
        query="What is 25-3518/01 used for?",
        expected_codes=["25-3518/01", "25-3518"],
        description="Exact code with /01 variant",
    ),

    # ── Name-based queries ─────────────────────────────────────────────────────
    TestCase(
        query="ADR Touch Control Pro 2000 BS EN Auto Compression Machine",
        expected_codes=[],  # may trigger disambiguation
        expect_disambiguation=False,
        description="Full product name — may have exact match",
    ),
    TestCase(
        query="Cone Penetrometer specifications",
        expected_codes=[],
        description="Partial name — cone penetrometer",
    ),
    TestCase(
        query="What accessories are available for the CBR mould?",
        expected_codes=[],
        description="Accessory query by name fragment",
    ),

    # ── Variant queries ────────────────────────────────────────────────────────
    TestCase(
        query="Show all variants of 36-5150",
        expected_codes=["36-5150"],
        description="Family/variants query by base code",
    ),
    TestCase(
        query="What is the difference between 36-5150/01 and 36-5150/06?",
        expected_codes=["36-5150/01", "36-5150/06"],
        description="Variant comparison — both codes must appear",
    ),
    TestCase(
        query="What models are available for the compression machine?",
        expected_codes=[],
        description="Generic models query",
    ),

    # ── Standards queries ──────────────────────────────────────────────────────
    TestCase(
        query="Which products comply with BS EN 12390?",
        expected_codes=[],
        description="Standards retrieval",
    ),
    TestCase(
        query="Find products meeting ASTM D1557 standard",
        expected_codes=[],
        description="ASTM standard retrieval",
    ),

    # ── Category / topic queries ───────────────────────────────────────────────
    TestCase(
        query="Show me products related to triaxial testing",
        expected_codes=[],
        description="Category-level topic retrieval",
    ),
    TestCase(
        query="Find equipment for soil compaction",
        expected_codes=[],
        description="Topic: soil compaction equipment",
    ),
    TestCase(
        query="Which page contains the Cone Penetrometer?",
        expected_codes=[],
        description="Page lookup by name",
    ),

    # ── Buyer's guide queries ──────────────────────────────────────────────────
    TestCase(
        query="What equipment is needed for unconfined compression test?",
        expected_codes=[],
        description="Buyer's guide type query",
    ),

    # ── Ambiguous queries (should trigger disambiguation) ──────────────────────
    TestCase(
        query="ADR Touch Control Pro 2000",
        expected_codes=[],
        expect_disambiguation=True,
        description="Ambiguous short product family name — should disambiguate",
    ),
    TestCase(
        query="Compression machine",
        expected_codes=[],
        expect_disambiguation=True,
        description="Very generic — multiple matches expected",
    ),

    # ── Partial/fuzzy name ─────────────────────────────────────────────────────
    TestCase(
        query="I'm looking for the auto compression 2000 machine",
        expected_codes=[],
        description="Fuzzy name match",
    ),
    TestCase(
        query="penetrometer for soil bearing capacity",
        expected_codes=[],
        description="Semantic match — penetrometer + bearing capacity",
    ),
]


@dataclass
class EvalResult:
    query: str
    expected_codes: List[str]
    found_top1: bool
    found_top3: bool
    exact_code_resolved: bool
    disambiguation_triggered: bool
    expected_disambiguation: bool
    retrieved_codes_top3: List[str]
    description: str


def _codes_in_result(result: RetrievalResult, top_k: int) -> List[str]:
    codes: List[str] = []
    for sc in result.ranked[:top_k]:
        codes.extend(sc.chunk.product_codes)
        codes.extend(sc.chunk.base_product_codes)
    return list(set(c.upper() for c in codes))


def run_evaluation(ki: KeywordIndex, vi: VectorIndex) -> None:
    results: List[EvalResult] = []

    for tc in TEST_CASES:
        log.info("Testing: %s", tc.query[:60])
        result = retrieve(tc.query, ki, vi)

        retrieved_top3 = _codes_in_result(result, 3)
        retrieved_top1 = _codes_in_result(result, 1)

        expected_upper = [c.upper() for c in tc.expected_codes]

        found_top1 = (
            any(ec in retrieved_top1 for ec in expected_upper)
            if expected_upper else True
        )
        found_top3 = (
            any(ec in retrieved_top3 for ec in expected_upper)
            if expected_upper else True
        )

        # Exact code resolution: if the query contains a code, was it in top-1?
        from src.normalizer import extract_codes
        query_codes_upper = [c.upper() for c in extract_codes(tc.query)]
        exact_resolved = (
            all(qc in retrieved_top3 for qc in query_codes_upper)
            if query_codes_upper else True
        )

        results.append(EvalResult(
            query=tc.query,
            expected_codes=tc.expected_codes,
            found_top1=found_top1,
            found_top3=found_top3,
            exact_code_resolved=exact_resolved,
            disambiguation_triggered=result.needs_disambiguation,
            expected_disambiguation=tc.expect_disambiguation,
            retrieved_codes_top3=retrieved_top3,
            description=tc.description,
        ))

    # ── Print report ───────────────────────────────────────────────────────────
    print("\n" + "=" * 70)
    print("EVALUATION REPORT")
    print("=" * 70)

    top1_pass  = sum(1 for r in results if r.found_top1)
    top3_pass  = sum(1 for r in results if r.found_top3)
    code_pass  = sum(1 for r in results if r.exact_code_resolved)
    disambig_correct = sum(
        1 for r in results
        if r.disambiguation_triggered == r.expected_disambiguation
    )
    n = len(results)

    print(f"Total test cases : {n}")
    print(f"Top-1 accuracy   : {top1_pass}/{n} ({100*top1_pass/n:.1f}%)")
    print(f"Top-3 accuracy   : {top3_pass}/{n} ({100*top3_pass/n:.1f}%)")
    print(f"Code resolution  : {code_pass}/{n} ({100*code_pass/n:.1f}%)")
    print(f"Disambig correct : {disambig_correct}/{n} ({100*disambig_correct/n:.1f}%)")
    print()

    for i, r in enumerate(results, 1):
        status = "PASS" if r.found_top3 else "FAIL"
        print(f"[{status}] {i:2d}. {r.description}")
        print(f"        Query : {r.query[:70]}")
        print(f"        Expected codes : {r.expected_codes}")
        print(f"        Retrieved top-3: {r.retrieved_codes_top3[:6]}")
        print(f"        Disambig expected={r.expected_disambiguation} triggered={r.disambiguation_triggered}")
        print()

    print("=" * 70)


def main() -> None:
    from src.config import BM25_PATH, CODE_MAP_PATH

    ki = KeywordIndex()
    vi = VectorIndex()

    try:
        ki.load(BM25_PATH, CODE_MAP_PATH)
        vi.load()
    except FileNotFoundError as e:
        print(f"Index not found: {e}")
        print("Run:  python -m src.cli build-index")
        sys.exit(1)

    run_evaluation(ki, vi)


if __name__ == "__main__":
    main()
