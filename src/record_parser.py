"""
Parse raw page text into structured ProductRecord objects.

Strategy:
  1. Classify each page (product_record, buyers_guide, category_intro, index_page, other).
  2. Detect product record boundaries using structural heuristics.
  3. Extract sub-sections: title, codes, standards, description, specs, accessories, spares, models.
  4. Merge multi-page records where appropriate.
"""
from __future__ import annotations

import hashlib
import re
from typing import List, Optional, Dict, Any

from src.config import EXTRACTED_RECORDS_PATH
from src.schema import ProductRecord, RawPage
from src.normalizer import (
    extract_codes,
    derive_base_codes,
    split_code,
    generate_aliases,
    normalise_text,
    PRODUCT_CODE_RE,
)
from src.utils import get_logger, write_jsonl, ensure_dirs

log = get_logger(__name__)

# ── Section header patterns ────────────────────────────────────────────────────
_PRODUCT_CODE_HEADER = re.compile(
    r"product\s+codes?\s*:?\s*([^\n]*)", re.IGNORECASE
)
_STANDARDS_HEADER = re.compile(
    r"(?:product\s+)?standards?\s*:?\s*([^\n]*)", re.IGNORECASE
)
_SPECS_HEADER = re.compile(
    r"specifications?\s*:?\s*", re.IGNORECASE
)
_ACCESSORIES_HEADER = re.compile(
    r"accessories\s*:?\s*", re.IGNORECASE
)
_SPARES_HEADER = re.compile(
    r"(?:spares?|consumables?)\s*(?:/\s*consumables?)?\s*:?\s*", re.IGNORECASE
)
_MODELS_HEADER = re.compile(
    r"(?:models?|ordering\s+information|available\s+models?)\s*:?\s*",
    re.IGNORECASE,
)
_GROUPED_STANDARDS = re.compile(
    r"grouped\s+product\s+standards?\s*:?\s*", re.IGNORECASE
)

# Patterns that signal non-product pages
_BUYERS_GUIDE_SIGNALS = re.compile(
    r"buyer'?s?\s+guide|selection\s+guide|guide\s+to\s+", re.IGNORECASE
)
_INDEX_SIGNALS = re.compile(
    r"^\s*(contents?|index)\s*$", re.IGNORECASE | re.MULTILINE
)
_CATEGORY_INTRO_SIGNALS = re.compile(
    r"^\s*[A-Z][A-Za-z\s&/,]+\s*$", re.MULTILINE
)

# A line looks like a product title if it is title-case with ≥3 words and ≤120 chars
_TITLE_LINE_RE = re.compile(
    r"^[A-Z][A-Za-z0-9\s\-&,./()]+$"
)

# Standards found inline in text
_INLINE_STANDARDS_RE = re.compile(
    r"\b(BS\s*EN\s*[\w\s:-]+|ASTM\s+[A-Z]\d+(?:-\d+)?|ISO\s+\d+(?:[-:]\d+)?|AASHTO\s+[A-Z\s]+\d+)",
    re.IGNORECASE,
)


def _record_id(page_start: int, name: Optional[str], codes: List[str], text_prefix: str = "") -> str:
    # Include text_prefix to disambiguate multiple products on the same page
    key = f"{page_start}|{name}|{'|'.join(codes)}|{text_prefix[:80]}"
    return "rec_" + hashlib.md5(key.encode()).hexdigest()[:12]


def _classify_page(text: str, codes: List[str]) -> str:
    if _BUYERS_GUIDE_SIGNALS.search(text):
        return "buyers_guide"
    if _INDEX_SIGNALS.search(text) and len(codes) < 3:
        return "index_page"
    # Very few product codes and short page → probably intro
    if len(codes) == 0 and len(text.strip()) < 600:
        return "category_intro"
    if len(codes) >= 1:
        return "product_record"
    return "other"


def _extract_section_after(text: str, header_re: re.Pattern) -> Optional[str]:
    """
    Return text between the first match of header_re and the next recognised header
    or end of text.
    """
    m = header_re.search(text)
    if not m:
        return None

    start = m.end()
    # Find next section header
    other_headers = [
        _PRODUCT_CODE_HEADER, _STANDARDS_HEADER, _SPECS_HEADER,
        _ACCESSORIES_HEADER, _SPARES_HEADER, _MODELS_HEADER, _GROUPED_STANDARDS,
    ]
    next_start = len(text)
    for hdr in other_headers:
        hm = hdr.search(text, start)
        if hm and hm.start() < next_start:
            next_start = hm.start()

    section = text[start:next_start].strip()
    return section if section else None


def _extract_title(lines: List[str]) -> Optional[str]:
    """
    Find the product title by looking for the line immediately before
    'Product Code:' — that is the most reliable signal in a product catalogue.
    Falls back to a heuristic scan if no 'Product Code:' header is present.
    """
    # Primary strategy: find line just before "Product Code:" header
    for i, line in enumerate(lines):
        if _PRODUCT_CODE_HEADER.match(line.strip()):
            # Walk backwards to find the nearest non-empty candidate line
            for j in range(i - 1, max(i - 5, -1), -1):
                candidate = lines[j].strip()
                if not candidate:
                    continue
                # Skip very short lines and obvious non-titles
                if len(candidate) < 5:
                    continue
                # Skip lines that look like page numbers, URLs, emails
                if re.match(r"^\d+$", candidate):
                    continue
                if "@" in candidate or "www." in candidate:
                    continue
                # Skip lines that are themselves section headers
                if any(
                    hdr.match(candidate)
                    for hdr in [_STANDARDS_HEADER, _SPECS_HEADER,
                                _ACCESSORIES_HEADER, _SPARES_HEADER, _MODELS_HEADER]
                ):
                    continue
                return candidate
            break

    # Fallback: find first plausible title-case line
    for line in lines[:20]:
        stripped = line.strip()
        if len(stripped) < 6:
            continue
        words = stripped.split()
        if len(words) < 2:
            continue
        if not stripped[0].isupper():
            continue
        if any(
            hdr.match(stripped)
            for hdr in [_PRODUCT_CODE_HEADER, _STANDARDS_HEADER, _SPECS_HEADER,
                        _ACCESSORIES_HEADER, _SPARES_HEADER, _MODELS_HEADER]
        ):
            continue
        if PRODUCT_CODE_RE.fullmatch(stripped.replace(" ", "")):
            continue
        return stripped
    return None


_STANDARDS_LINE_RE = re.compile(
    r"\b(BS|EN|ASTM|ISO|AASHTO|DIN|NF|UNI|NBN|UNE|JIS|GB)\b",
    re.IGNORECASE,
)


def _extract_standards(text: str) -> List[str]:
    standards: List[str] = []
    seen: set = set()

    # From "Product Standards:" section header — only keep lines that look like standards
    m = _STANDARDS_HEADER.search(text)
    if m:
        # Look at the inline group(1) first (codes on the same line as the header)
        inline = m.group(1).strip()
        if inline and _STANDARDS_LINE_RE.search(inline):
            for part in re.split(r"[,;]", inline):
                s = part.strip()
                if s and s not in seen:
                    seen.add(s)
                    standards.append(s)

        # Then scan subsequent lines until we hit a non-standards line
        after_pos = m.end()
        for line in text[after_pos:].splitlines():
            stripped = line.strip().rstrip(",;")
            if not stripped:
                continue
            # Stop as soon as we hit a line that doesn't look like a standard
            if not _STANDARDS_LINE_RE.search(stripped):
                break
            if stripped not in seen:
                seen.add(stripped)
                standards.append(stripped)

    # Also harvest inline standard references from the whole text
    for match in _INLINE_STANDARDS_RE.finditer(text):
        s = match.group(0).strip().rstrip(",;")
        if s and s not in seen:
            seen.add(s)
            standards.append(s)

    return standards[:20]


def _build_searchable_text(record: Dict[str, Any]) -> str:
    parts = []
    if record.get("product_name"):
        parts.append(record["product_name"])
    parts.extend(record.get("product_codes", []))
    parts.extend(record.get("aliases", []))
    parts.extend(record.get("standards", []))
    if record.get("description"):
        parts.append(record["description"])
    if record.get("specifications_text"):
        parts.append(record["specifications_text"])
    if record.get("accessories_text"):
        parts.append(record["accessories_text"])
    return " ".join(parts)


def parse_page(raw_page: RawPage, category: Optional[str] = None) -> ProductRecord:
    text = raw_page.text
    lines = text.splitlines()
    codes = extract_codes(text)
    content_type = _classify_page(text, codes)

    product_name = _extract_title(lines) if content_type == "product_record" else None

    # Extract codes from explicit 'Product Code:' headers first
    code_header_match = _PRODUCT_CODE_HEADER.search(text)
    if code_header_match:
        code_line = code_header_match.group(1)
        header_codes = extract_codes(code_line)
        # Prepend header codes (they are most reliable)
        for c in reversed(header_codes):
            if c not in codes:
                codes.insert(0, c)

    base_codes = derive_base_codes(codes)
    suffixes = []
    for c in codes:
        _, suf = split_code(c)
        if suf and suf not in suffixes:
            suffixes.append(suf)

    standards = _extract_standards(text)
    description = _extract_section_after(text, re.compile(r"description\s*:?\s*", re.IGNORECASE))
    specs_text = _extract_section_after(text, _SPECS_HEADER)
    accessories_text = _extract_section_after(text, _ACCESSORIES_HEADER)
    spares_text = _extract_section_after(text, _SPARES_HEADER)
    models_text = _extract_section_after(text, _MODELS_HEADER)

    aliases = generate_aliases(product_name, codes)

    rec_dict: Dict[str, Any] = {
        "record_id": _record_id(raw_page.page_number, product_name, codes, text[:80]),
        "content_type": content_type,
        "page_start": raw_page.page_number,
        "page_end": raw_page.page_number,
        "category": category,
        "subcategory": None,
        "product_name": product_name,
        "normalized_product_name": normalise_text(product_name) if product_name else None,
        "product_codes": codes,
        "base_product_codes": base_codes,
        "variant_suffixes": suffixes,
        "standards": standards,
        "description": description,
        "specifications_text": specs_text,
        "accessories_text": accessories_text,
        "spares_text": spares_text,
        "models_text": models_text,
        "aliases": aliases,
        "raw_text": text,
        "searchable_text": "",
    }
    rec_dict["searchable_text"] = _build_searchable_text(rec_dict)
    return ProductRecord(**rec_dict)


def _split_page_into_product_blocks(text: str) -> List[str]:
    """
    Split a page that contains multiple 'Product Code:' headers into
    one text block per product. Lines before the first code header are
    prepended to each block as context (they contain the category/subcategory).
    """
    # Find all positions of "Product Code:" lines
    splits: List[int] = []
    for m in _PRODUCT_CODE_HEADER.finditer(text):
        # Walk back to find the title line (the non-empty line before this header)
        before = text[:m.start()]
        lines_before = before.splitlines()
        title_start = m.start()
        for i in range(len(lines_before) - 1, max(len(lines_before) - 6, -1), -1):
            ln = lines_before[i].strip()
            if ln and not re.match(r"^\d+$", ln) and "@" not in ln and "www." not in ln:
                # This is the title line — start the block here
                title_start = sum(len(l) + 1 for l in lines_before[:i])
                break
        splits.append(title_start)

    if len(splits) <= 1:
        return [text]

    # Preamble = everything before the first product title
    preamble = text[:splits[0]].strip()

    blocks: List[str] = []
    for i, start in enumerate(splits):
        end = splits[i + 1] if i + 1 < len(splits) else len(text)
        block = text[start:end].strip()
        if preamble:
            block = preamble + "\n\n" + block
        blocks.append(block)

    return blocks


def parse_pages(raw_pages: List[RawPage]) -> List[ProductRecord]:
    """
    Parse all raw pages into ProductRecord objects.

    Pages with multiple 'Product Code:' headers are split into sub-records,
    one per product, so that each product gets its own properly titled record.
    """
    records: List[ProductRecord] = []
    current_category: Optional[str] = None

    for raw_page in raw_pages:
        text = raw_page.text.strip()
        if not text:
            continue

        # Detect category-header pages (short, title-case, no codes)
        codes_on_page = extract_codes(text)
        if (
            len(codes_on_page) == 0
            and len(text) < 400
            and len(text.splitlines()) < 8
        ):
            first_line = next((ln.strip() for ln in text.splitlines() if ln.strip()), "")
            if first_line and first_line[0].isupper() and len(first_line) > 4:
                current_category = first_line
                log.debug("Category detected: %s (page %d)", current_category, raw_page.page_number)

        # Count 'Product Code:' occurrences to decide whether to split
        code_header_count = len(_PRODUCT_CODE_HEADER.findall(text))

        if code_header_count > 1 and not _BUYERS_GUIDE_SIGNALS.search(text):
            # Multi-product page — split into individual blocks
            blocks = _split_page_into_product_blocks(text)
            log.debug("Page %d split into %d product blocks", raw_page.page_number, len(blocks))
            for block in blocks:
                sub_page = RawPage(
                    page_number=raw_page.page_number,
                    text=block,
                    char_count=len(block),
                    has_tables=raw_page.has_tables,
                )
                record = parse_page(sub_page, category=current_category)
                records.append(record)
        else:
            record = parse_page(raw_page, category=current_category)
            records.append(record)

    log.info("Parsed %d records from %d pages", len(records), len(raw_pages))
    _log_summary(records)
    return records


def _log_summary(records: List[ProductRecord]) -> None:
    from collections import Counter
    counts = Counter(r.content_type for r in records)
    for ct, n in counts.most_common():
        log.info("  content_type=%-20s count=%d", ct, n)


def save_records(records: List[ProductRecord], out_path=EXTRACTED_RECORDS_PATH) -> None:
    ensure_dirs(out_path.parent)
    count = write_jsonl(out_path, records)
    log.info("Saved %d records to %s", count, out_path)


def run(raw_pages: List[RawPage]) -> List[ProductRecord]:
    records = parse_pages(raw_pages)
    save_records(records)
    return records
