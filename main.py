from __future__ import annotations

import argparse
import sys
from pathlib import Path

import uvicorn

from rag_agent.ingest import ingest_pdfs
from rag_agent.retriever import Retriever
from rag_agent.llm import generate_answer
from rag_agent import config


def cmd_ingest(paths: list[str]) -> None:
	if not paths:
		print("Provide at least one PDF path.")
		sys.exit(1)
	stats = ingest_pdfs([Path(p) for p in paths])
	print(f"Ingestion complete: {stats}")


def cmd_ask(question: str) -> None:
	try:
		retriever = Retriever()
	except Exception as e:
		print(f"Error: {e}")
		sys.exit(1)
	results = retriever.search(question)
	contexts = [
		{"text": r.text, "score": r.score, "source_path": r.source_path, "page_number": r.page_number, "chunk_index": r.chunk_index}
		for r in results
	]
	answer = generate_answer(question, contexts)
	print("\nAnswer:\n" + answer["answer"]) 
	print("\nSources:")
	for s in answer["sources"]:
		print(f"- {s['source']} (page {s['page']}), score={s['score']:.3f}")


def cmd_runserver(host: str, port: int, reload: bool) -> None:
	uvicorn.run("app:app", host=host, port=port, reload=reload)


def main() -> None:
	parser = argparse.ArgumentParser(description="RAG PDF QA Agent")
	sub = parser.add_subparsers(dest="command", required=True)

	p_ingest = sub.add_parser("ingest", help="Ingest one or more PDF files")
	p_ingest.add_argument("paths", nargs="+", help="Absolute paths to PDF files")

	p_ask = sub.add_parser("ask", help="Ask a question against the knowledge base")
	p_ask.add_argument("question", help="Your question")

	p_run = sub.add_parser("runserver", help="Run FastAPI server")
	p_run.add_argument("--host", default="0.0.0.0")
	p_run.add_argument("--port", type=int, default=8000)
	p_run.add_argument("--reload", action="store_true")

	args = parser.parse_args()
	config.ensure_directories()

	if args.command == "ingest":
		cmd_ingest(args.paths)
	elif args.command == "ask":
		cmd_ask(args.question)
	elif args.command == "runserver":
		cmd_runserver(args.host, args.port, args.reload)
	else:
		parser.print_help()


if __name__ == "__main__":
	main()

