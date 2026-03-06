"""
Ingestion pipeline: extract text from PDF/DOCX, chunk it, embed it,
and persist the vectors in a ChromaDB collection.
"""

from __future__ import annotations

import logging
import shutil
from pathlib import Path
from typing import Optional

import fitz  # PyMuPDF
from docx import Document as DocxDocument
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document

import config
from pipelines.vectorstore import add_documents_to_store

logger = logging.getLogger(__name__)


# ── File Parsing ─────────────────────────────────────────────────

def extract_text_from_pdf(file_path: str | Path) -> str:
    """Extract full text from a PDF using PyMuPDF."""
    text_parts: list[str] = []
    with fitz.open(str(file_path)) as doc:
        for page in doc:
            text_parts.append(page.get_text())
    return "\n".join(text_parts)


def extract_text_from_docx(file_path: str | Path) -> str:
    """Extract full text from a DOCX using python-docx."""
    doc = DocxDocument(str(file_path))
    return "\n".join(para.text for para in doc.paragraphs if para.text.strip())


def extract_text(file_path: str | Path) -> str:
    """Route to the correct parser based on file extension."""
    path = Path(file_path)
    ext = path.suffix.lower()

    if ext == ".pdf":
        return extract_text_from_pdf(path)
    elif ext == ".docx":
        return extract_text_from_docx(path)
    else:
        raise ValueError(
            f"Unsupported file type: {ext}. "
            f"Supported: {config.SUPPORTED_EXTENSIONS}"
        )


# ── Chunking ─────────────────────────────────────────────────────

def chunk_text(
    text: str,
    source: str,
    chunk_size: Optional[int] = None,
    chunk_overlap: Optional[int] = None,
) -> list[Document]:
    """
    Split extracted text into LangChain Documents with metadata.
    Chunk size and overlap are configurable via parameters or config.
    """
    chunk_size = chunk_size or config.CHUNK_SIZE
    chunk_overlap = chunk_overlap or config.CHUNK_OVERLAP

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        length_function=len,
        separators=["\n\n", "\n", ". ", " ", ""],
    )

    chunks = splitter.create_documents(
        texts=[text],
        metadatas=[{"source": source}],
    )

    for i, chunk in enumerate(chunks):
        chunk.metadata["chunk_index"] = i

    logger.info("Split '%s' into %d chunks (size=%d, overlap=%d)",
                source, len(chunks), chunk_size, chunk_overlap)
    return chunks


# ── Top-Level Ingestion Entrypoint ───────────────────────────────

def save_uploaded_file(file_path: str | Path) -> Path:
    """Copy an uploaded file into the project uploads directory."""
    src = Path(file_path)
    dest = config.UPLOAD_DIR / src.name
    shutil.copy2(src, dest)
    logger.info("Saved uploaded file to %s", dest)
    return dest


def ingest_document(
    file_path: str | Path,
    chunk_size: Optional[int] = None,
    chunk_overlap: Optional[int] = None,
) -> dict:
    """
    Full ingestion pipeline:
      1. Copy file to uploads/
      2. Extract text
      3. Chunk the text
      4. Embed & store in ChromaDB

    Returns a summary dict with ingestion statistics.
    """
    path = Path(file_path)

    if path.suffix.lower() not in config.SUPPORTED_EXTENSIONS:
        raise ValueError(
            f"Unsupported file type: {path.suffix}. "
            f"Supported: {config.SUPPORTED_EXTENSIONS}"
        )

    saved_path = save_uploaded_file(path)
    text = extract_text(saved_path)

    if not text.strip():
        raise ValueError(f"No extractable text found in {path.name}")

    chunks = chunk_text(
        text,
        source=path.name,
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
    )

    add_documents_to_store(chunks)

    return {
        "filename": path.name,
        "characters": len(text),
        "chunks": len(chunks),
        "chunk_size": chunk_size or config.CHUNK_SIZE,
        "chunk_overlap": chunk_overlap or config.CHUNK_OVERLAP,
        "status": "success",
    }


# ── CLI helper ───────────────────────────────────────────────────

if __name__ == "__main__":
    import sys

    logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")

    if len(sys.argv) < 2:
        print("Usage: python -m pipelines.ingestion <file_path> [chunk_size] [chunk_overlap]")
        sys.exit(1)

    fpath = sys.argv[1]
    cs = int(sys.argv[2]) if len(sys.argv) > 2 else None
    co = int(sys.argv[3]) if len(sys.argv) > 3 else None

    result = ingest_document(fpath, chunk_size=cs, chunk_overlap=co)
    print(f"\nIngestion complete: {result}")
