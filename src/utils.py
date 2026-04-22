"""
Shared utilities: logging setup, JSONL I/O, directory helpers.
"""
from __future__ import annotations

import json
import logging
import sys
from pathlib import Path
from typing import Any, Iterable, Iterator, List

from src.config import LOG_LEVEL


def get_logger(name: str) -> logging.Logger:
    logger = logging.getLogger(name)
    if not logger.handlers:
        handler = logging.StreamHandler(sys.stderr)
        handler.setFormatter(
            logging.Formatter("%(asctime)s [%(levelname)s] %(name)s: %(message)s",
                              datefmt="%H:%M:%S")
        )
        logger.addHandler(handler)
        logger.setLevel(getattr(logging, LOG_LEVEL.upper(), logging.INFO))
    return logger


def ensure_dirs(*paths: Path) -> None:
    for p in paths:
        p.mkdir(parents=True, exist_ok=True)


def write_jsonl(path: Path, records: Iterable[Any]) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with path.open("w", encoding="utf-8") as f:
        for rec in records:
            if hasattr(rec, "model_dump"):
                f.write(json.dumps(rec.model_dump(), ensure_ascii=False) + "\n")
            else:
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")
            count += 1
    return count


def read_jsonl(path: Path) -> List[dict]:
    if not path.exists():
        raise FileNotFoundError(f"JSONL file not found: {path}")
    records = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def iter_jsonl(path: Path) -> Iterator[dict]:
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                yield json.loads(line)


def truncate(text: str, max_chars: int = 300) -> str:
    if len(text) <= max_chars:
        return text
    return text[:max_chars] + "…"
