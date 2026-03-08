"""
Shared vector store and embedding model access.

Provides singleton access to the HuggingFace embedding model and the
persistent ChromaDB collection. Used by both the ingestion and retrieval
pipelines to avoid duplicating setup logic.
"""

from __future__ import annotations

import logging
import shutil

from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_classic.retrievers.multi_vector import MultiVectorRetriever
from langchain_classic.storage import LocalFileStore

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

def get_docstore():
    """Return a handle to the persistent local file store for parent documents."""
    import json
    from langchain_classic.storage import EncoderBackedStore

    fs = LocalFileStore(str(config.DOCSTORE_DIR))

    def serialize_doc(doc: Document) -> bytes:
        return json.dumps({
            "page_content": doc.page_content, 
            "metadata": doc.metadata
        }).encode("utf-8")
        
    def deserialize_doc(b: bytes) -> Document:
        d = json.loads(b.decode("utf-8"))
        return Document(page_content=d["page_content"], metadata=d["metadata"])

    return EncoderBackedStore(
        store=fs,
        key_encoder=lambda x: x,
        value_serializer=serialize_doc,
        value_deserializer=deserialize_doc
    )

def get_retriever() -> MultiVectorRetriever:
    """Return the parent-child multi-vector retriever."""
    return MultiVectorRetriever(
        vectorstore=get_vectorstore(),
        docstore=get_docstore(),
        id_key="doc_id",
        search_kwargs={"k": config.RETRIEVAL_TOP_K}
    )

def add_documents_to_store(parent_docs: list[Document], child_docs: list[Document]) -> MultiVectorRetriever:
    """Embed child documents and link them to parent documents via docstore."""
    retriever = get_retriever()
    
    # Store the parent documents in the key-value store mapping doc_id -> Document
    doc_ids = [doc.metadata["doc_id"] for doc in parent_docs]
    retriever.docstore.mset(list(zip(doc_ids, parent_docs)))
    
    # Embed and add the child documents to the vectorstore
    retriever.vectorstore.add_documents(child_docs)
    
    logger.info(
        "Added %d parents and %d children to vector store '%s'",
        len(parent_docs), len(child_docs),
        config.CHROMA_COLLECTION_NAME,
    )
    return retriever


def clear_vectorstore() -> None:
    """Delete current collection and docstore so the next use starts with empty stores."""
    import chromadb
    client = chromadb.PersistentClient(path=str(config.VECTORSTORE_DIR))
    try:
        client.delete_collection(name=config.CHROMA_COLLECTION_NAME)
        logger.info("Deleted collection '%s'", config.CHROMA_COLLECTION_NAME)
    except Exception as e:
        logger.warning("Could not delete collection (may not exist): %s", e)
        
    try:
        if config.DOCSTORE_DIR.exists():
            shutil.rmtree(config.DOCSTORE_DIR)
            config.DOCSTORE_DIR.mkdir()
            logger.info("Cleared parent document store")
    except Exception as e:
        logger.warning("Could not clear document store: %s", e)
