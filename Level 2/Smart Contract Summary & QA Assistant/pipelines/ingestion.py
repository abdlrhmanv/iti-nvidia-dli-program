"""
Ingestion pipeline: extract text from PDF/DOCX, chunk it, embed it,
and persist the vectors in a ChromaDB collection.
"""

from __future__ import annotations

import logging
import shutil
from pathlib import Path
import uuid

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
) -> tuple[list[Document], list[Document]]:
    """
    Split extracted text into Parent Documents (large) and Child Documents (small).
    Links them together via a shared 'doc_id' metadata field.
    """
    parent_splitter = RecursiveCharacterTextSplitter(
        chunk_size=config.PARENT_CHUNK_SIZE,
        chunk_overlap=config.PARENT_CHUNK_OVERLAP,
        length_function=len,
        separators=["\n\n", "\n", ". ", " ", ""],
    )
    
    child_splitter = RecursiveCharacterTextSplitter(
        chunk_size=config.CHILD_CHUNK_SIZE,
        chunk_overlap=config.CHILD_CHUNK_OVERLAP,
        length_function=len,
        separators=["\n\n", "\n", ". ", " ", ""],
    )

    parent_docs = parent_splitter.create_documents(
        texts=[text],
        metadatas=[{"source": source}],
    )

    child_docs = []
    
    for i, p_doc in enumerate(parent_docs):
        # Assign a unique ID to the parent
        doc_id = str(uuid.uuid4())
        p_doc.metadata["doc_id"] = doc_id
        p_doc.metadata["chunk_index"] = i
        
        # Split the parent into smaller children
        sub_docs = child_splitter.split_documents([p_doc])
        for sub_doc in sub_docs:
            # The child must carry the exact same 'doc_id' map back to the parent
            sub_doc.metadata["doc_id"] = doc_id
            child_docs.append(sub_doc)

    logger.info("Split '%s' into %d parent chunks and %d child chunks",
                source, len(parent_docs), len(child_docs))
                
    return parent_docs, child_docs


# ── Top-Level Ingestion Entrypoint ───────────────────────────────

def save_uploaded_file(file_path: str | Path) -> Path:
    """Copy an uploaded file into the project uploads directory."""
    src = Path(file_path)
    dest = config.UPLOAD_DIR / src.name
    try:
        shutil.copy2(src, dest)
        logger.info("Saved uploaded file to %s", dest)
    except shutil.SameFileError:
        logger.info("File already exists at %s", dest)
    return dest


def ingest_document(
    file_path: str | Path,
) -> dict:
    """
    Full ingestion pipeline (Parent-Child Strategy):
      1. Copy file to uploads/
      2. Extract text
      3. Create Parent (large) and Child (small) chunks
      4. Persist Parents in Docstore, embed Children in Chroma

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

    parent_docs, child_docs = chunk_text(
        text,
        source=path.name,
    )

    add_documents_to_store(parent_docs, child_docs)

    return {
        "filename": path.name,
        "characters": len(text),
        "parent_chunks": len(parent_docs),
        "child_chunks": len(child_docs),
        "chunk_size": config.PARENT_CHUNK_SIZE,
        "status": "success",
    }


# ── CLI helper ───────────────────────────────────────────────────

if __name__ == "__main__":
    import sys

    logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")

    if len(sys.argv) < 2:
        print("Usage: python -m pipelines.ingestion <file_path>")
        sys.exit(1)
    fpath = sys.argv[1]

    result = ingest_document(fpath)
    print(f"\nIngestion complete: {result}")
