## RAG PDF QA Agent

An end-to-end Retrieval Augmented Generation (RAG) app to ingest PDF files, build a vector index, and answer questions grounded in your documents. Ships with:

- FastAPI backend with a minimal web UI
- PDF ingestion and chunking
- Embeddings via `sentence-transformers/all-MiniLM-L6-v2`
- Vector search via FAISS
- Optional generation via OpenAI (falls back to concise extractive answers if no key is provided)

### 1) Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
```

Optional (recommended) for better answers:

```bash
export OPENAI_API_KEY=YOUR_KEY
```

### 2) Run the app

```bash
uvicorn app:app --reload --host 0.0.0.0 --port 8000
```

Open the UI at `http://localhost:8000/`.

### 3) Usage

1. Upload one or more PDFs under "Build knowledge base" and click Ingest. This creates/updates a FAISS index under `data/`.
2. Ask a question under "Ask a question". The app retrieves the most relevant chunks and uses OpenAI (if available) to compose an answer. If no API key is configured, it returns a concise, grounded extract from the sources.

### 4) CLI (optional)

```bash
python main.py ingest /absolute/path/to/doc1.pdf /absolute/path/to/doc2.pdf
python main.py ask "What is the warranty policy?"
python main.py runserver --port 8000
```

### 5) Project structure

```
AI_Agent/
  app.py                # FastAPI app + minimal UI
  main.py               # CLI entrypoint
  rag_agent/
    __init__.py
    config.py           # Paths and tunables
    ingest.py           # PDF parsing, chunking, indexing
    retriever.py        # FAISS search
    llm.py              # OpenAI integration + fallback
  static/
    index.html          # Minimal frontend
  data/                 # Index + metadata storage
```

### 6) Notes

- The index uses cosine similarity (via inner product on normalized vectors).
- Chunking: by paragraphs then sliding windows; configurable in `rag_agent/config.py`.
- Safe answering: the model is instructed to say it doesn't know if the answer isn't in the provided context.

### 7) Troubleshooting

- If `faiss` install fails on Apple Silicon, ensure you are on Python 3.10–3.11 and using `faiss-cpu`:
  - `pip install --no-cache-dir faiss-cpu`
- Large PDFs: ingestion may take time on first run while models download.
- If you prefer local generation, you can integrate with Ollama by editing `rag_agent/llm.py` to call your local model instead of OpenAI.

