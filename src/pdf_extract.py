"""
PDF text extraction using pymupdf (fitz) with pdfplumber fallback.

Outputs: artifacts/raw_pages.jsonl — one JSON object per page.
"""
from __future__ import annotations

import re
from pathlib import Path
from typing import List

from src.config import PDF_PATH, RAW_PAGES_PATH
from src.schema import RawPage
from src.utils import get_logger, write_jsonl, ensure_dirs

log = get_logger(__name__)


def extract_pages(pdf_path: Path = PDF_PATH) -> List[RawPage]:
    if not pdf_path.exists():
        raise FileNotFoundError(
            f"PDF not found at {pdf_path}. "
            "Place your catalogue PDF at data/product_catalogue.pdf "
            "or set PDF_FILENAME in .env"
        )

    log.info("Extracting text from %s", pdf_path)

    try:
        pages = _extract_with_pymupdf(pdf_path)
        log.info("pymupdf extracted %d pages", len(pages))
    except ImportError:
        log.warning("pymupdf not available, falling back to pdfplumber")
        pages = _extract_with_pdfplumber(pdf_path)
        log.info("pdfplumber extracted %d pages", len(pages))

    return pages


def _extract_with_pymupdf(pdf_path: Path) -> List[RawPage]:
    import fitz  # type: ignore

    doc = fitz.open(str(pdf_path))
    pages: List[RawPage] = []

    for page_idx in range(len(doc)):
        page = doc[page_idx]
        text = page.get_text("text")
        text = _clean_text(text)

        # Try to detect tables by presence of grid-like patterns
        has_tables = _heuristic_has_table(text)

        pages.append(RawPage(
            page_number=page_idx + 1,
            text=text,
            char_count=len(text),
            has_tables=has_tables,
        ))

    doc.close()
    return pages


def _extract_with_pdfplumber(pdf_path: Path) -> List[RawPage]:
    import pdfplumber  # type: ignore

    pages: List[RawPage] = []

    with pdfplumber.open(str(pdf_path)) as pdf:
        for page_idx, page in enumerate(pdf.pages):
            text = page.extract_text() or ""
            table_text: str | None = None
            has_tables = False

            tables = page.extract_tables()
            if tables:
                has_tables = True
                rows = []
                for table in tables:
                    for row in table:
                        clean_row = [str(cell or "").strip() for cell in row]
                        rows.append(" | ".join(clean_row))
                table_text = "\n".join(rows)

            combined = text
            if table_text:
                combined = text + "\n\n[TABLE]\n" + table_text

            combined = _clean_text(combined)

            pages.append(RawPage(
                page_number=page_idx + 1,
                text=combined,
                char_count=len(combined),
                has_tables=has_tables,
                table_text=table_text,
            ))

    return pages


def _clean_text(text: str) -> str:
    # Normalize unicode dashes and quotes
    text = text.replace("–", "-").replace("—", "-")
    text = text.replace("‘", "'").replace("’", "'")
    text = text.replace("“", '"').replace("”", '"')
    # Collapse excessive blank lines
    text = re.sub(r"\n{3,}", "\n\n", text)
    # Strip trailing spaces per line
    lines = [ln.rstrip() for ln in text.splitlines()]
    return "\n".join(lines).strip()


def _heuristic_has_table(text: str) -> bool:
    # Simple heuristic: multiple lines with consistent pipe or tab separators
    pipe_lines = sum(1 for ln in text.splitlines() if "|" in ln)
    return pipe_lines >= 3


def save_pages(pages: List[RawPage], out_path: Path = RAW_PAGES_PATH) -> None:
    ensure_dirs(out_path.parent)
    count = write_jsonl(out_path, pages)
    log.info("Saved %d raw pages to %s", count, out_path)


def run(pdf_path: Path = PDF_PATH) -> List[RawPage]:
    pages = extract_pages(pdf_path)
    save_pages(pages)
    return pages
