from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Dict, List, Tuple

import faiss
import numpy as np
from sentence_transformers import SentenceTransformer

from . import config


@dataclass
class RetrievedChunk:
	text: str
	score: float
	source_path: str
	page_number: int
	chunk_index: int


class Retriever:
	"""Load FAISS index/metadata and provide vector similarity search."""

	def __init__(self) -> None:
		config.ensure_directories()
		if not config.INDEX_PATH.exists() or not config.METADATA_PATH.exists():
			raise RuntimeError("No index/metadata found. Please ingest PDFs first.")
		self.index: faiss.Index = faiss.read_index(str(config.INDEX_PATH))
		self.metadata: List[Dict] = []
		with open(config.METADATA_PATH, "r", encoding="utf-8") as f:
			for line in f:
				self.metadata.append(json.loads(line))
		self.model: SentenceTransformer = SentenceTransformer(config.EMBEDDING_MODEL_NAME)

	def _embed_query(self, query: str) -> np.ndarray:
		emb = self.model.encode([query], convert_to_numpy=True, normalize_embeddings=False)
		n = np.linalg.norm(emb, axis=1, keepdims=True) + 1e-10
		return (emb / n).astype(np.float32)

	def search(self, query: str, top_k: int | None = None) -> List[RetrievedChunk]:
		k = top_k or config.TOP_K
		query_vec = self._embed_query(query)
		scores, indices = self.index.search(query_vec, k)
		scores = scores.flatten().tolist()
		indices = indices.flatten().tolist()
		results: List[RetrievedChunk] = []
		for score, idx in zip(scores, indices):
			if idx < 0 or idx >= len(self.metadata):
				continue
			m = self.metadata[idx]
			results.append(
				RetrievedChunk(
					text=m["text"],
					score=float(score),
					source_path=m["source_path"],
					page_number=int(m["page_number"]),
					chunk_index=int(m["chunk_index"]),
				)
			)
		return results

