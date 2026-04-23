"""
Text and code normalisation plus alias generation.
"""
from __future__ import annotations

import re
import unicodedata
from typing import List, Optional, Tuple

# Product code pattern: DD-DDDD or DD-DDDD/DD (also handles longer suffixes)
PRODUCT_CODE_RE = re.compile(r"\b(\d{2,3}-\d{3,5})(?:/(\d{2,3}))?\b")

# Standards markers to optionally strip from titles
STANDARDS_RE = re.compile(
    r"\b(BS|EN|ASTM|ISO|AASHTO|DIN|NF|UNI|NBN|UNE|JIS|GB)\b[\s\w/-]*",
    re.IGNORECASE,
)

# Trailing power / supply references  e.g. "230V" "110/230V"
POWER_RE = re.compile(r"\b\d{2,3}(?:/\d{2,3})?V\b", re.IGNORECASE)


def normalise_text(text: str) -> str:
    """Lowercase, unicode normalise, collapse whitespace."""
    text = unicodedata.normalize("NFKC", text)
    text = text.lower()
    text = re.sub(r"[\s ]+", " ", text)
    return text.strip()


def normalise_code(code: str) -> str:
    """Uppercase + strip extra spaces in a product code string."""
    return re.sub(r"\s+", "", code.upper())


def extract_codes(text: str) -> List[str]:
    """Return all product codes found in text, normalised."""
    return [normalise_code(m.group(0)) for m in PRODUCT_CODE_RE.finditer(text)]


def split_code(code: str) -> Tuple[str, Optional[str]]:
    """
    Split '36-5150/01' -> ('36-5150', '01').
    Split '36-5150'    -> ('36-5150', None).
    """
    m = PRODUCT_CODE_RE.match(code.strip())
    if m:
        return normalise_code(m.group(1)), m.group(2)
    return normalise_code(code), None


def derive_base_codes(codes: List[str]) -> List[str]:
    """Return unique base codes (without variant suffix)."""
    seen: set[str] = set()
    result: List[str] = []
    for c in codes:
        base, _ = split_code(c)
        if base not in seen:
            seen.add(base)
            result.append(base)
    return result


def generate_aliases(
    product_name: Optional[str],
    codes: List[str],
) -> List[str]:
    """
    Generate a conservative set of searchable aliases for a product.
    Conservative means: no aliases that are so generic they'd match many products.
    """
    aliases: List[str] = []
    seen: set[str] = set()

    def add(a: str) -> None:
        a = a.strip()
        if a and a not in seen and len(a) > 2:
            seen.add(a)
            aliases.append(a)

    # Code-based aliases
    for code in codes:
        add(code)
        base, suffix = split_code(code)
        add(base)
        if suffix:
            add(f"{base}/{suffix}")

    if not product_name:
        return aliases

    # Full title (original case)
    add(product_name)

    # Lowercase title
    name_lower = normalise_text(product_name)
    add(name_lower)

    # Strip trailing standards clause (e.g., "BS EN Auto Compression Machine" → "Auto Compression Machine")
    stripped_standards = STANDARDS_RE.sub("", product_name).strip()
    if stripped_standards and stripped_standards != product_name:
        add(stripped_standards)
        add(normalise_text(stripped_standards))

    # Strip power references
    stripped_power = POWER_RE.sub("", product_name).strip()
    if stripped_power and stripped_power != product_name:
        add(stripped_power)
        add(normalise_text(stripped_power))

    # Remove both standards and power
    both_stripped = POWER_RE.sub("", STANDARDS_RE.sub("", product_name)).strip()
    if both_stripped and both_stripped != product_name:
        add(both_stripped)
        add(normalise_text(both_stripped))

    return aliases
