.PHONY: setup pull-models index ask test ui clean help

# ── Setup ──────────────────────────────────────────────────────────────────────
setup:
	@echo "Creating virtual environment and installing dependencies..."
	python3 -m venv .venv
	.venv/bin/pip install --upgrade pip
	.venv/bin/pip install -r requirements.txt
	cp -n .env.example .env 2>/dev/null || true
	@echo ""
	@echo "Setup complete. Activate with: source .venv/bin/activate"
	@echo "Then pull models: make pull-models"

# ── Ollama model pulling ───────────────────────────────────────────────────────
pull-models:
	@echo "Pulling Ollama models (requires Ollama running)..."
	ollama pull llama3.1:8b
	ollama pull nomic-embed-text
	@echo "Models pulled successfully."

pull-models-alt:
	@echo "Pulling alternative models..."
	ollama pull qwen2.5:7b
	ollama pull mxbai-embed-large

# ── Indexing ───────────────────────────────────────────────────────────────────
index:
	@echo "Building index from data/product_catalogue.pdf ..."
	python -m src.cli build-index

index-pdf:
	@if [ -z "$(PDF)" ]; then echo "Usage: make index-pdf PDF=path/to/file.pdf"; exit 1; fi
	python -m src.cli build-index --pdf $(PDF)

# ── Querying ───────────────────────────────────────────────────────────────────
ask:
	@if [ -z "$(Q)" ]; then echo "Usage: make ask Q=\"your question here\""; exit 1; fi
	python -m src.cli ask "$(Q)"

inspect:
	@if [ -z "$(CODE)" ]; then echo "Usage: make inspect CODE=36-5150/01"; exit 1; fi
	python -m src.cli inspect-record "$(CODE)"

debug:
	@if [ -z "$(Q)" ]; then echo "Usage: make debug Q=\"your query\""; exit 1; fi
	python -m src.cli debug-retrieval "$(Q)"

# ── Web UI ─────────────────────────────────────────────────────────────────────
ui:
	streamlit run src/app.py

# ── Evaluation ─────────────────────────────────────────────────────────────────
eval:
	python -m src.evaluate

# ── Tests ──────────────────────────────────────────────────────────────────────
test:
	pytest tests/ -v

test-cov:
	pytest tests/ -v --cov=src --cov-report=term-missing

# ── Clean ──────────────────────────────────────────────────────────────────────
clean-artifacts:
	rm -f artifacts/*.jsonl

clean-index:
	rm -rf index_store/chroma index_store/metadata

clean: clean-artifacts clean-index
	@echo "Cleaned artifacts and indexes."

# ── Help ───────────────────────────────────────────────────────────────────────
help:
	@echo ""
	@echo "Catalogue RAG — available commands:"
	@echo ""
	@echo "  make setup          Install dependencies and copy .env"
	@echo "  make pull-models    Pull Ollama generation + embedding models"
	@echo "  make index          Build index from data/product_catalogue.pdf"
	@echo "  make index-pdf PDF=<path>  Build index from a specific PDF"
	@echo "  make ask Q=\"...\"    Ask a question"
	@echo "  make inspect CODE=36-5150/01  Inspect a product code"
	@echo "  make debug Q=\"...\" Show retrieval debug scores"
	@echo "  make ui             Launch Streamlit web UI"
	@echo "  make eval           Run evaluation benchmark"
	@echo "  make test           Run unit tests"
	@echo "  make clean          Remove artifacts and index"
	@echo ""
