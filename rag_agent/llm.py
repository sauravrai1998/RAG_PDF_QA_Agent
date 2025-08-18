from __future__ import annotations

import os
from typing import List, Tuple

from . import config


SYSTEM_PROMPT = (
	"You are a helpful assistant that answers questions using only the provided context. "
	"If the answer cannot be found in the context, say you don't know succinctly. "
	"Cite sources as (page N) when relevant. Keep answers concise."
)


def _format_context(chunks: List[Tuple[str, str]]) -> str:
	"""Format a list of (text, source_label) into a prompt context block."""
	lines = []
	for text, label in chunks:
		lines.append(f"[{label}]\n{text}")
	return "\n\n".join(lines)


def generate_answer(query: str, contexts: List[dict]) -> dict:
	"""Generate an answer grounded in retrieved contexts. If OPENAI_API_KEY is set, use OpenAI.

	Otherwise, return a concise extractive answer by quoting the most relevant chunk.
	"""
	key = config.get_openai_api_key()
	ordered = sorted(contexts, key=lambda c: c.get("score", 0.0), reverse=True)
	context_pairs = []
	for c in ordered:
		label = f"{os.path.basename(c['source_path'])} (page {c['page_number']})"
		context_pairs.append((c["text"], label))

	if not key:
		# Fallback: extractive concise answer
		top = ordered[0] if ordered else None
		answer = top["text"][:600] + ("..." if len(top["text"]) > 600 else "") if top else "I couldn't find relevant context."
		return {
			"answer": answer,
			"used_model": "extractive-fallback",
			"sources": [
				{
					"source": os.path.basename(c["source_path"]),
					"page": c["page_number"],
					"score": c.get("score", 0.0),
				}
				for c in ordered
			],
		}

	# OpenAI path
	try:
		from openai import OpenAI  # type: ignore
		client = OpenAI(api_key=key)
		context_block = _format_context(context_pairs)
		messages = [
			{"role": "system", "content": SYSTEM_PROMPT},
			{
				"role": "user",
				"content": f"Context:\n\n{context_block}\n\nQuestion: {query}",
			},
		]
		completion = client.chat.completions.create(
			model="gpt-4o-mini",
			messages=messages,
			temperature=0.1,
			max_tokens=500,
		)
		text = completion.choices[0].message.content or ""
		sources = [
			{"source": os.path.basename(c["source_path"]), "page": c["page_number"], "score": c.get("score", 0.0)}
			for c in ordered
		]
		return {"answer": text.strip(), "used_model": "gpt-4o-mini", "sources": sources}
	except Exception as e:  # graceful fallback
		top = ordered[0] if ordered else None
		answer = top["text"][:600] + ("..." if top and len(top["text"]) > 600 else "") if top else "I couldn't find relevant context."
		return {
			"answer": answer,
			"used_model": f"fallback-due-to-error: {type(e).__name__}",
			"sources": [
				{
					"source": os.path.basename(c["source_path"]),
					"page": c["page_number"],
					"score": c.get("score", 0.0),
				}
				for c in ordered
			],
		}

