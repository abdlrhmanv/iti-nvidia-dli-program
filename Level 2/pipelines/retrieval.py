"""
Retrieval pipeline: query the ChromaDB vector store for semantically
similar chunks and return them as ranked LangChain Documents.
"""

from __future__ import annotations

import logging
from typing import Optional

from langchain_core.documents import Document

import config
from pipelines.ingestion import get_vectorstore

logger = logging.getLogger(__name__)


def retrieve_chunks(
    query: str,
    top_k: Optional[int] = None,
) -> list[Document]:
    """
    Retrieve the most relevant document chunks for a user query.

    Uses cosine similarity (via normalized embeddings) against the
    persistent ChromaDB collection.

    Returns LangChain Documents with .page_content and .metadata
    (source filename, chunk_index, relevance score).
    """
    top_k = top_k or config.RETRIEVAL_TOP_K
    store = get_vectorstore()

    results = store.similarity_search_with_relevance_scores(query, k=top_k)

    docs: list[Document] = []
    for doc, score in results:
        doc.metadata["relevance_score"] = round(score, 4)
        docs.append(doc)

    logger.info(
        "Retrieved %d chunks for query (top_k=%d): '%.80s...'",
        len(docs), top_k, query,
    )
    return docs


def format_context(docs: list[Document]) -> str:
    """
    Build a numbered context block from retrieved chunks for injection
    into the LLM prompt. Each chunk is tagged with its source and index
    so the LLM can cite them.
    """
    if not docs:
        return "No relevant context found."

    parts: list[str] = []
    for i, doc in enumerate(docs, 1):
        source = doc.metadata.get("source", "unknown")
        chunk_idx = doc.metadata.get("chunk_index", "?")
        parts.append(
            f"[Source {i}: {source}, chunk {chunk_idx}]\n"
            f"{doc.page_content}"
        )
    return "\n\n---\n\n".join(parts)


# ── CLI helper ───────────────────────────────────────────────────

if __name__ == "__main__":
    import sys

    logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")

    if len(sys.argv) < 2:
        print("Usage: python -m pipelines.retrieval <query> [top_k]")
        sys.exit(1)

    query = sys.argv[1]
    k = int(sys.argv[2]) if len(sys.argv) > 2 else None

    chunks = retrieve_chunks(query, top_k=k)
    for i, chunk in enumerate(chunks, 1):
        score = chunk.metadata.get("relevance_score", "N/A")
        source = chunk.metadata.get("source", "unknown")
        print(f"\n── Chunk {i} (score={score}, source={source}) ──")
        print(chunk.page_content[:300])
