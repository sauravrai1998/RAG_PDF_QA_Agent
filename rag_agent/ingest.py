from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import faiss
import numpy as np
from pypdf import PdfReader
from sentence_transformers import SentenceTransformer
from tqdm import tqdm

from . import config


@dataclass
class ChunkMetadata:
	chunk_id: int
	text: str
	source_path: str
	page_number: int
	chunk_index: int


def _read_pdf_text(pdf_path: Path) -> List[str]:
	"""Extract text per page from a PDF file."""
	reader = PdfReader(str(pdf_path))
	pages: List[str] = []
	for page in reader.pages:
		pages.append(page.extract_text() or "")
	return pages


def _split_into_paragraphs(text: str) -> List[str]:
	paragraphs = [p.strip() for p in text.split("\n\n")]
	return [p for p in paragraphs if p]


def _sliding_window(paragraphs: List[str], chunk_size: int, overlap: int) -> List[str]:
	"""Create chunks using a sliding window over concatenated paragraphs."""
	joined = "\n\n".join(paragraphs)
	chunks: List[str] = []
	start = 0
	length = len(joined)
	while start < length and len(chunks) < config.MAX_CHUNKS_PER_PAGE:
		end = min(start + chunk_size, length)
		chunk = joined[start:end]
		chunks.append(chunk)
		if end == length:
			break
		start = max(0, end - overlap)
	return chunks


def _normalize_embeddings(embeddings: np.ndarray) -> np.ndarray:
	"""L2-normalize for cosine similarity with FAISS inner product index."""
	norms = np.linalg.norm(embeddings, axis=1, keepdims=True) + 1e-10
	return embeddings / norms


def _load_embedding_model() -> SentenceTransformer:
	return SentenceTransformer(config.EMBEDDING_MODEL_NAME)


def _embed_texts(model: SentenceTransformer, texts: List[str]) -> np.ndarray:
	emb = model.encode(texts, show_progress_bar=True, convert_to_numpy=True, batch_size=64, normalize_embeddings=False)
	return _normalize_embeddings(emb)


def _load_index_and_metadata() -> Tuple[faiss.Index, List[Dict]]:
	"""Load existing FAISS index and metadata if available, otherwise create new."""
	if config.INDEX_PATH.exists() and config.METADATA_PATH.exists():
		index = faiss.read_index(str(config.INDEX_PATH))
		metadata: List[Dict] = []
		with open(config.METADATA_PATH, "r", encoding="utf-8") as f:
			for line in f:
				metadata.append(json.loads(line))
		return index, metadata
	# Create empty index; infer dimension from model
	model = _load_embedding_model()
	dim = model.get_sentence_embedding_dimension()
	index = faiss.IndexFlatIP(dim)
	return index, []


def _append_to_store(index: faiss.Index, metadata_store: List[Dict], embeddings: np.ndarray, chunk_metas: List[ChunkMetadata]) -> Tuple[faiss.Index, List[Dict]]:
	"""Append new embeddings and metadata to the existing index and store."""
	start_id = len(metadata_store)
	index.add(embeddings.astype(np.float32))
	for i, meta in enumerate(chunk_metas):
		record = {
			"chunk_id": start_id + i,
			"text": meta.text,
			"source_path": meta.source_path,
			"page_number": meta.page_number,
			"chunk_index": meta.chunk_index,
		}
		metadata_store.append(record)
	return index, metadata_store


def _save_index_and_metadata(index: faiss.Index, metadata_store: List[Dict]) -> None:
	faiss.write_index(index, str(config.INDEX_PATH))
	with open(config.METADATA_PATH, "w", encoding="utf-8") as f:
		for rec in metadata_store:
			f.write(json.dumps(rec, ensure_ascii=False) + "\n")


def ingest_pdfs(pdf_paths: Iterable[Path]) -> Dict[str, int]:
	"""Ingest PDF files, updating the FAISS index and metadata store.

	Returns statistics about the ingestion process.
	"""
	config.ensure_directories()
	index, metadata_store = _load_index_and_metadata()
	model = _load_embedding_model()

	all_new_chunks: List[str] = []
	all_new_metas: List[ChunkMetadata] = []

	for path in pdf_paths:
		path = Path(path)
		if not path.exists() or path.suffix.lower() != ".pdf":
			continue
		pages = _read_pdf_text(path)
		for page_num, text in enumerate(pages, start=1):
			paragraphs = _split_into_paragraphs(text)
			if not paragraphs:
				continue
			chunks = _sliding_window(paragraphs, chunk_size=config.CHUNK_SIZE, overlap=config.CHUNK_OVERLAP)
			for chunk_idx, chunk in enumerate(chunks):
				all_new_chunks.append(chunk)
				all_new_metas.append(
					ChunkMetadata(
						chunk_id=-1,
						text=chunk,
						source_path=str(path.resolve()),
						page_number=page_num,
						chunk_index=chunk_idx,
					)
				)

	if not all_new_chunks:
		return {"added_chunks": 0, "total_chunks": len(metadata_store)}

	embeddings = _embed_texts(model, all_new_chunks)
	index, metadata_store = _append_to_store(index, metadata_store, embeddings, all_new_metas)
	_save_index_and_metadata(index, metadata_store)

	return {"added_chunks": len(all_new_chunks), "total_chunks": len(metadata_store)}

