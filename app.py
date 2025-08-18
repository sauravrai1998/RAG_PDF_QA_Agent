from __future__ import annotations

import os
from pathlib import Path
from typing import List

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, HTMLResponse
from fastapi.staticfiles import StaticFiles

from rag_agent import config
from rag_agent.ingest import ingest_pdfs
from rag_agent.retriever import Retriever
from rag_agent.llm import generate_answer


app = FastAPI(title="RAG PDF QA Agent", version="1.0.0")

app.add_middleware(
	CORSMiddleware,
	allow_origins=["*"],
	allow_credentials=True,
	allow_methods=["*"],
	allow_headers=["*"],
)


config.ensure_directories()
static_dir = Path(__file__).parent / "static"
app.mount("/static", StaticFiles(directory=static_dir), name="static")

_retriever: Retriever | None = None


@app.on_event("startup")
def _load_retriever_on_startup() -> None:
	global _retriever
	try:
		_retriever = Retriever()
	except Exception:
		_retriever = None


@app.get("/", response_class=HTMLResponse)
def index() -> HTMLResponse:
	index_file = static_dir / "index.html"
	return HTMLResponse(index_file.read_text(encoding="utf-8"))


@app.post("/ingest")
async def ingest(files: List[UploadFile] = File(...)):
	if not files:
		raise HTTPException(status_code=400, detail="No files uploaded")
	paths: List[Path] = []
	for f in files:
		if not f.filename.lower().endswith(".pdf"):
			raise HTTPException(status_code=400, detail=f"Unsupported file type: {f.filename}")
		dst = config.UPLOADS_DIR / f.filename
		with open(dst, "wb") as out:
			out.write(await f.read())
		paths.append(dst)

	stats = ingest_pdfs(paths)
	# Reload retriever after ingestion
	global _retriever
	try:
		_retriever = Retriever()
	except Exception as e:
		raise HTTPException(status_code=500, detail=f"Failed to initialize retriever: {e}")
	return {"message": "Ingestion complete", **stats}


@app.post("/ask")
async def ask(payload: dict):
	global _retriever
	if _retriever is None:
		try:
			_retriever = Retriever()
		except Exception:
			raise HTTPException(status_code=400, detail="No knowledge base found. Ingest PDFs first.")
	question = payload.get("question", "").strip()
	if not question:
		raise HTTPException(status_code=400, detail="Question is required")
	results = _retriever.search(question)
	contexts = [
		{"text": r.text, "score": r.score, "source_path": r.source_path, "page_number": r.page_number, "chunk_index": r.chunk_index}
		for r in results
	]
	answer = generate_answer(question, contexts)
	return {"question": question, **answer}

