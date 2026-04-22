"""
Pydantic data models for product records and chunks.
"""
from __future__ import annotations

from typing import List, Optional
from pydantic import BaseModel, Field


class ProductRecord(BaseModel):
    record_id: str
    content_type: str  # product_record, buyers_guide, category_intro, index_page, other

    page_start: int
    page_end: int

    category: Optional[str] = None
    subcategory: Optional[str] = None

    product_name: Optional[str] = None
    normalized_product_name: Optional[str] = None

    product_codes: List[str] = Field(default_factory=list)
    base_product_codes: List[str] = Field(default_factory=list)
    variant_suffixes: List[str] = Field(default_factory=list)

    standards: List[str] = Field(default_factory=list)
    description: Optional[str] = None
    specifications_text: Optional[str] = None
    accessories_text: Optional[str] = None
    spares_text: Optional[str] = None
    models_text: Optional[str] = None

    aliases: List[str] = Field(default_factory=list)
    searchable_text: str = ""

    raw_text: str = ""


class ChunkRecord(BaseModel):
    chunk_id: str
    record_id: str
    chunk_type: str  # title, summary, specs, accessories, full_record, buyers_guide_row, etc.

    page_start: int
    page_end: int

    product_name: Optional[str] = None
    product_codes: List[str] = Field(default_factory=list)
    base_product_codes: List[str] = Field(default_factory=list)
    category: Optional[str] = None
    content_type: str = "other"

    text: str = ""
    searchable_text: str = ""


class RawPage(BaseModel):
    page_number: int
    text: str
    char_count: int
    has_tables: bool = False
    table_text: Optional[str] = None
