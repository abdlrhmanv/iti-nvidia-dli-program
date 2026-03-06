"""
pipelines — public API for the Smart Contract Assistant.

Import the main entry points here so consumers (app.py, scripts, tests)
can do:
    from pipelines import ingest_document, answer_question, retrieve_chunks
"""

from pipelines.ingestion import ingest_document
from pipelines.retrieval import retrieve_chunks, format_context
from pipelines.llm_pipeline import answer_question
from pipelines.vectorstore import get_vectorstore, get_embedding_model

__all__ = [
    "ingest_document",
    "retrieve_chunks",
    "format_context",
    "answer_question",
    "get_vectorstore",
    "get_embedding_model",
]
