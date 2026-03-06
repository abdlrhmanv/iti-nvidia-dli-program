"""
Shared vector store and embedding model access.

Provides singleton access to the HuggingFace embedding model and the
persistent ChromaDB collection. Used by both the ingestion and retrieval
pipelines to avoid duplicating setup logic.
"""

from __future__ import annotations

import logging

from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_huggingface import HuggingFaceEmbeddings

import config

logger = logging.getLogger(__name__)

_embedding_model: HuggingFaceEmbeddings | None = None


def get_embedding_model() -> HuggingFaceEmbeddings:
    """Lazy-load the embedding model (singleton)."""
    global _embedding_model
    if _embedding_model is None:
        _embedding_model = HuggingFaceEmbeddings(
            model_name=config.EMBEDDING_MODEL_NAME,
            model_kwargs={"device": config.EMBEDDING_DEVICE},
            encode_kwargs={"normalize_embeddings": True},
        )
        logger.info(
            "Loaded embedding model: %s on %s",
            config.EMBEDDING_MODEL_NAME,
            config.EMBEDDING_DEVICE,
        )
    return _embedding_model


def get_vectorstore() -> Chroma:
    """Return a handle to the persistent Chroma vector store."""
    return Chroma(
        collection_name=config.CHROMA_COLLECTION_NAME,
        embedding_function=get_embedding_model(),
        persist_directory=str(config.VECTORSTORE_DIR),
    )


def add_documents_to_store(docs: list[Document]) -> Chroma:
    """Embed documents and add them to the persistent vector store."""
    store = get_vectorstore()
    store.add_documents(docs)
    logger.info(
        "Added %d chunks to vector store '%s'",
        len(docs),
        config.CHROMA_COLLECTION_NAME,
    )
    return store
