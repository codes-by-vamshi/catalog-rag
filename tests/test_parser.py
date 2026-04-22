"""Tests for record_parser.py"""
import pytest
from src.schema import RawPage
from src.record_parser import parse_page, _classify_page, _extract_standards, _extract_title


SAMPLE_PRODUCT_TEXT = """ADR Touch Control Pro 2000 BS EN Auto Compression Machine

Product Codes: 36-5150/01, 36-5150/06

Product Standards: BS EN 12390-4, ASTM C39

Specifications:
Load capacity: 3000 kN
Frame stiffness: 11 MN/mm
Display: 7 inch touch screen

Accessories:
- Compression platens 36-5151/01
- Calibration weight set 36-9999

Spares/Consumables:
- Hydraulic oil 36-0001
"""

SAMPLE_BUYERS_GUIDE = """Buyer's Guide to Compression Testing

This guide helps you select the right compression machine.

Product Code  Description          Standards
36-5150/01    Pro 2000             BS EN 12390-4
36-5150/06    Pro 2000 ASTM        ASTM C39
"""

SAMPLE_CATEGORY_PAGE = """Concrete Testing

Our range of concrete testing equipment covers compression,
flexural, and tensile testing.
"""


class TestClassifyPage:
    def test_product_page(self):
        ct = _classify_page(SAMPLE_PRODUCT_TEXT, ["36-5150/01", "36-5150/06"])
        assert ct == "product_record"

    def test_buyers_guide(self):
        ct = _classify_page(SAMPLE_BUYERS_GUIDE, ["36-5150/01", "36-5150/06"])
        assert ct == "buyers_guide"

    def test_category_intro(self):
        ct = _classify_page(SAMPLE_CATEGORY_PAGE, [])
        assert ct == "category_intro"


class TestExtractTitle:
    def test_finds_title(self):
        lines = SAMPLE_PRODUCT_TEXT.splitlines()
        title = _extract_title(lines)
        assert title == "ADR Touch Control Pro 2000 BS EN Auto Compression Machine"

    def test_no_title_in_empty(self):
        assert _extract_title([]) is None

    def test_ignores_section_headers(self):
        lines = ["Product Codes: 36-5150/01", "Real Title Here"]
        title = _extract_title(lines)
        # Should not pick up "Product Codes:" line
        assert title != "Product Codes: 36-5150/01"


class TestExtractStandards:
    def test_finds_standards_from_header(self):
        standards = _extract_standards(SAMPLE_PRODUCT_TEXT)
        assert any("BS EN 12390" in s for s in standards)

    def test_finds_astm(self):
        standards = _extract_standards(SAMPLE_PRODUCT_TEXT)
        assert any("ASTM" in s for s in standards)


class TestParsePage:
    def make_page(self, text: str, page_num: int = 1) -> RawPage:
        return RawPage(page_number=page_num, text=text, char_count=len(text))

    def test_product_record_type(self):
        page = self.make_page(SAMPLE_PRODUCT_TEXT)
        record = parse_page(page)
        assert record.content_type == "product_record"

    def test_extracts_codes(self):
        page = self.make_page(SAMPLE_PRODUCT_TEXT)
        record = parse_page(page)
        assert "36-5150/01" in record.product_codes or "36-5150/06" in record.product_codes

    def test_extracts_base_codes(self):
        page = self.make_page(SAMPLE_PRODUCT_TEXT)
        record = parse_page(page)
        assert "36-5150" in record.base_product_codes

    def test_extracts_title(self):
        page = self.make_page(SAMPLE_PRODUCT_TEXT)
        record = parse_page(page)
        assert record.product_name is not None
        assert "Compression" in record.product_name or "ADR" in record.product_name

    def test_extracts_specifications(self):
        page = self.make_page(SAMPLE_PRODUCT_TEXT)
        record = parse_page(page)
        assert record.specifications_text is not None
        assert "3000" in record.specifications_text or "kN" in record.specifications_text

    def test_extracts_accessories(self):
        page = self.make_page(SAMPLE_PRODUCT_TEXT)
        record = parse_page(page)
        assert record.accessories_text is not None

    def test_page_numbers_preserved(self):
        page = self.make_page(SAMPLE_PRODUCT_TEXT, page_num=42)
        record = parse_page(page)
        assert record.page_start == 42

    def test_searchable_text_populated(self):
        page = self.make_page(SAMPLE_PRODUCT_TEXT)
        record = parse_page(page)
        assert len(record.searchable_text) > 50

    def test_buyers_guide_classified(self):
        page = self.make_page(SAMPLE_BUYERS_GUIDE)
        record = parse_page(page)
        assert record.content_type == "buyers_guide"

    def test_aliases_generated(self):
        page = self.make_page(SAMPLE_PRODUCT_TEXT)
        record = parse_page(page)
        assert len(record.aliases) > 0
        assert any("36-5150" in a for a in record.aliases)
