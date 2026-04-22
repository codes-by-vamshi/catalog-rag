"""
Answer generation via Ollama.

Builds a prompt from retrieved chunks, sends it to the local LLM,
and formats the response with grounding metadata.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import List, Optional

import requests

from src.config import OLLAMA_BASE_URL, GENERATION_MODEL, PROMPTS_DIR
from src.retriever import RetrievalResult, ScoredChunk
from src.utils import get_logger, truncate

log = get_logger(__name__)

_MAX_CONTEXT_CHARS = 6000


def _load_prompt(name: str) -> str:
    path = PROMPTS_DIR / name
    if path.exists():
        return path.read_text(encoding="utf-8").strip()
    # Inline fallback (so the system works even without the prompts/ dir)
    return _INLINE_PROMPTS.get(name, "")


_INLINE_PROMPTS: dict = {
    "system_prompt.txt": (
        "You are a product catalogue assistant. Your job is to answer questions "
        "about products, specifications, standards, accessories, and models. "
        "Answer ONLY from the provided context. "
        "Never invent specifications, codes, or standards not present in the context. "
        "If the context does not contain enough information, say so explicitly. "
        "Always include the product code and page number in your answer when available."
    ),
    "answer_prompt.txt": (
        "Answer the user's question using ONLY the retrieved product information below.\n\n"
        "Rules:\n"
        "- Prioritise product codes over title-only matching.\n"
        "- If comparing variants, only compare those explicitly shown in context.\n"
        "- Always cite the page number and product code.\n"
        "- If unsure, say 'I do not have enough information'.\n"
        "- Do NOT invent data.\n\n"
        "Retrieved context:\n{context}\n\n"
        "Question: {question}\n\n"
        "Answer:"
    ),
    "disambiguation_prompt.txt": (
        "The query matches multiple products. List the candidates clearly.\n"
        "Ask the user to specify the exact product code or variant."
    ),
}


def format_context(ranked: List[ScoredChunk], max_chars: int = _MAX_CONTEXT_CHARS) -> str:
    parts: List[str] = []
    total = 0
    for i, sc in enumerate(ranked):
        chunk = sc.chunk
        header = (
            f"[Source {i + 1}] "
            f"Page {chunk.page_start}"
            + (f"-{chunk.page_end}" if chunk.page_end != chunk.page_start else "")
            + f" | Type: {chunk.chunk_type}"
        )
        if chunk.product_name:
            header += f" | Product: {chunk.product_name}"
        if chunk.product_codes:
            header += f" | Codes: {', '.join(chunk.product_codes)}"

        body = chunk.text.strip()
        block = f"{header}\n{body}\n"

        if total + len(block) > max_chars and parts:
            break
        parts.append(block)
        total += len(block)

    return "\n---\n".join(parts)


def format_disambiguation(candidates: List[ScoredChunk]) -> str:
    lines = [
        "I found multiple close matches for your query. "
        "Please specify the product code or variant you are looking for:\n"
    ]
    for i, sc in enumerate(candidates, 1):
        chunk = sc.chunk
        name = chunk.product_name or "(unnamed)"
        codes = ", ".join(chunk.product_codes) if chunk.product_codes else "no code"
        page = chunk.page_start
        lines.append(f"  {i}. {name} — {codes} — page {page}")

    lines.append(
        "\nPlease re-ask with the specific product code (e.g. '36-5150/01') "
        "to get a precise answer."
    )
    return "\n".join(lines)


def generate_answer(
    question: str,
    result: RetrievalResult,
) -> str:
    if result.needs_disambiguation:
        return format_disambiguation(result.disambiguation_candidates)

    if not result.ranked:
        return (
            "I could not find any matching products in the catalogue for your query. "
            "Try using a specific product code (e.g. '36-5150/01') or a more precise name."
        )

    context = format_context(result.ranked)
    answer_template = _load_prompt("answer_prompt.txt")
    system_prompt = _load_prompt("system_prompt.txt")

    user_message = answer_template.format(context=context, question=question)

    log.debug("Sending query to Ollama model: %s", GENERATION_MODEL)

    try:
        llm_answer = _call_ollama(system_prompt, user_message)
    except Exception as e:
        log.error("Ollama generation failed: %s", e)
        llm_answer = (
            "[LLM generation failed — showing retrieved evidence only]\n"
            + context[:2000]
        )

    # Append grounding footer
    footer = _build_grounding_footer(result.ranked)
    return f"{llm_answer}\n\n{footer}"


def _call_ollama(system: str, user: str) -> str:
    payload = {
        "model": GENERATION_MODEL,
        "messages": [
            {"role": "system", "content": system},
            {"role": "user",   "content": user},
        ],
        "stream": False,
        "options": {
            "temperature": 0.1,
            "num_predict": 1024,
        },
    }
    resp = requests.post(
        f"{OLLAMA_BASE_URL}/api/chat",
        json=payload,
        timeout=120,
    )
    if resp.status_code != 200:
        raise RuntimeError(f"Ollama /api/chat returned {resp.status_code}: {resp.text[:300]}")

    data = resp.json()
    return data["message"]["content"].strip()


def _build_grounding_footer(ranked: List[ScoredChunk]) -> str:
    seen_records: set = set()
    lines = ["---", "**Sources:**"]
    for sc in ranked:
        chunk = sc.chunk
        rid = chunk.record_id
        if rid in seen_records:
            continue
        seen_records.add(rid)

        name = chunk.product_name or "(unnamed)"
        codes = ", ".join(chunk.product_codes) if chunk.product_codes else "N/A"
        page = chunk.page_start
        lines.append(f"- **{name}** | Code: `{codes}` | Page: {page}")

    return "\n".join(lines)
