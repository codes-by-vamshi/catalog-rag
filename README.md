# Product Catalogue RAG System

A fully local, production-style RAG system for answering questions about a large product catalogue PDF.

- **Local-only**: Ollama for generation and embeddings. No paid APIs.
- **Catalogue-aware**: Exact product code matching, variant grouping, structured extraction.
- **Hybrid retrieval**: Exact code lookup + BM25 + vector search + product-aware reranking.
- **Disambiguation**: Never guesses between similar products — asks for clarification.
- **Grounded answers**: Every answer cites page number, product code, and evidence.

---

## Quick Start

### 1. Install dependencies

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env
```

Or use Make:

```bash
make setup
source .venv/bin/activate
```

### 2. Start Ollama

Install Ollama from https://ollama.com then:

```bash
ollama serve          # starts Ollama (or use Docker below)
```

Or with Docker:

```bash
docker-compose up -d ollama
```

### 3. Pull models

```bash
ollama pull llama3.1:8b
ollama pull nomic-embed-text
```

Or:

```bash
make pull-models
```

### 4. Place your PDF

```bash
cp /path/to/your/catalogue.pdf data/product_catalogue.pdf
```

### 5. Build the index

```bash
python -m src.cli build-index
```

Or:

```bash
make index
```

This will:
1. Extract all pages from the PDF
2. Parse product records with structured extraction
3. Create typed chunks (specs, accessories, models, etc.)
4. Build BM25 keyword index + exact code map
5. Build ChromaDB vector index with Ollama embeddings

Intermediate artifacts are saved to `artifacts/` for debugging.

---

## Usage

### CLI

```bash
# Ask a question
python -m src.cli ask "What are the specifications of 36-5150/01?"

# Ask about variants
python -m src.cli ask "Show all variants of 36-5150"

# Compare variants
python -m src.cli ask "What is the difference between 36-5150/01 and 36-5150/06?"

# Find accessories
python -m src.cli ask "What accessories are available for the CBR mould?"

# Inspect all chunks for a product code
python -m src.cli inspect-record 36-5150

# Debug retrieval scores for a query
python -m src.cli debug-retrieval "cone penetrometer"
```

### Make shortcuts

```bash
make ask Q="What standards does 36-5150/01 comply with?"
make inspect CODE=36-5150/01
make debug Q="compression machine"
```

### Web UI

```bash
streamlit run src/app.py
# or
make ui
```

Open http://localhost:8501 in your browser.

---

## Configuration

Copy `.env.example` to `.env` and edit:

| Variable | Default | Description |
|---|---|---|
| `PDF_FILENAME` | `product_catalogue.pdf` | Filename inside `data/` |
| `OLLAMA_BASE_URL` | `http://localhost:11434` | Ollama server URL |
| `GENERATION_MODEL` | `llama3.1:8b` | LLM for answer generation |
| `EMBEDDING_MODEL` | `nomic-embed-text` | Embedding model |
| `TOP_K_FINAL` | `6` | Number of chunks passed to LLM |
| `DISAMBIG_THRESHOLD` | `0.15` | Score spread that triggers disambiguation |
| `LOG_LEVEL` | `INFO` | Logging verbosity |

---

## Project Structure

```
catalog-rag/
├── data/
│   └── product_catalogue.pdf          ← place your PDF here
├── artifacts/
│   ├── raw_pages.jsonl                ← extracted page text
│   ├── extracted_records.jsonl        ← parsed product records
│   ├── chunks.jsonl                   ← typed chunks
│   └── debug_matches.jsonl            ← retrieval debug output
├── index_store/
│   ├── chroma/                        ← ChromaDB vector index
│   └── metadata/                      ← BM25 index + code map JSON
├── prompts/
│   ├── system_prompt.txt
│   ├── answer_prompt.txt
│   └── disambiguation_prompt.txt
├── src/
│   ├── config.py                      ← central config (reads .env)
│   ├── schema.py                      ← Pydantic models
│   ├── utils.py                       ← JSONL I/O, logging
│   ├── pdf_extract.py                 ← PDF → raw pages
│   ├── record_parser.py               ← raw pages → product records
│   ├── normalizer.py                  ← code/text normalisation + aliases
│   ├── chunker.py                     ← records → typed chunks
│   ├── embeddings.py                  ← Ollama / sentence-transformers
│   ├── keyword_index.py               ← BM25 + exact code map
│   ├── vector_index.py                ← ChromaDB vector index
│   ├── retriever.py                   ← multi-stage retrieval pipeline
│   ├── reranker.py                    ← product-aware scoring
│   ├── answerer.py                    ← LLM answer generation
│   ├── cli.py                         ← CLI entry point
│   ├── app.py                         ← Streamlit web UI
│   └── evaluate.py                    ← evaluation benchmark
└── tests/
    ├── test_normalization.py
    ├── test_parser.py
    ├── test_retrieval.py
    └── test_disambiguation.py
```

---

## How It Works

### Extraction Pipeline

1. **PDF extraction** (`pdf_extract.py`): Uses `pymupdf` (fallback: `pdfplumber`) to extract page text. Cleans unicode, collapses whitespace, detects tables.

2. **Record parsing** (`record_parser.py`): Classifies each page as `product_record`, `buyers_guide`, `category_intro`, `index_page`, or `other`. Extracts product codes, titles, standards, specs, accessories, spares, and models using regex + structural heuristics.

3. **Normalisation** (`normalizer.py`): Normalises codes (e.g. `36-5150/01`), splits into base code (`36-5150`) + variant (`01`), generates conservative aliases.

4. **Chunking** (`chunker.py`): Creates typed chunks per record — `title_and_description`, `specifications`, `accessories_and_spares`, `models_and_codes`, `full_record`. Buyer's guide pages get row-level chunks.

### Indexing

- **BM25** (`keyword_index.py`): Tokenised lexical search over all chunk text. Plus a separate exact code map (`code → [chunk_ids]`) for O(1) code lookups.
- **Vector** (`vector_index.py`): ChromaDB with cosine similarity, embeddings via Ollama (`nomic-embed-text`) or fallback to `sentence-transformers`.

### Retrieval

1. **Query analysis**: Detect product codes, base codes, comparison/variant/accessory intent.
2. **Exact code lookup**: If a code is in the query, retrieve it directly (highest priority).
3. **BM25 + vector search**: Merge results from both indexes.
4. **Reranking** (`reranker.py`): Score candidates with exact code match (+1.0), base code match (+0.6), name match (+0.4), content-type preference, chunk-type preference.
5. **Disambiguation**: If top results are too close and differ by product code, return candidates instead of guessing.

### Answer Generation

- Sends top-K chunks as context to Ollama.
- Strict system prompt: no fabrication, always cite codes and pages.
- Appends grounding footer: product name, code, page for each source.

---

## Running Tests

```bash
make test
# or
pytest tests/ -v
```

Tests cover: code normalisation, alias generation, record parsing, BM25 lookup, reranking, and disambiguation logic. Tests use in-memory indexes — no Ollama needed.

---

## Evaluation

```bash
make eval
# or
python -m src.evaluate
```

Runs 20 test queries covering exact codes, partial names, variants, accessories, standards, category retrieval, and ambiguous cases. Reports:
- Top-1 and Top-3 record accuracy
- Exact code resolution accuracy
- Disambiguation trigger correctness

---

## Troubleshooting

**`PDF not found`**
→ Place your PDF at `data/product_catalogue.pdf` or set `PDF_FILENAME` in `.env`.

**`BM25 index not found`**
→ Run `python -m src.cli build-index` first.

**`Ollama embedding failed`**
→ The system auto-falls back to `sentence-transformers/all-MiniLM-L6-v2`. Or run `ollama pull nomic-embed-text` and restart.

**`Ollama /api/chat failed`**
→ Check `ollama serve` is running. Verify `GENERATION_MODEL` in `.env` matches a pulled model.

**Poor retrieval quality**
→ Check `artifacts/extracted_records.jsonl` for extraction quality. Run `python -m src.cli debug-retrieval "your query"` to see scores.

**Disambiguation fires too often**
→ Lower `DISAMBIG_THRESHOLD` in `.env` (default 0.15).

**Disambiguation never fires**
→ Raise `DISAMBIG_THRESHOLD`.

---

## Alternative Models

If `llama3.1:8b` is too slow:

```bash
ollama pull qwen2.5:7b
# then in .env:
GENERATION_MODEL=qwen2.5:7b
```

If `nomic-embed-text` is unavailable:

```bash
ollama pull mxbai-embed-large
# then in .env:
EMBEDDING_MODEL=mxbai-embed-large
```

Or leave `EMBEDDING_MODEL` as is and the system will use `sentence-transformers` automatically.
