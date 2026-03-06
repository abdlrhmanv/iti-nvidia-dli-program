# Phase 1: Environment Setup & Ingestion Pipeline

## Overview

Phase 1 establishes the project foundation: a Python environment, a structured repository, and a complete document ingestion pipeline. By the end of this phase, the system can accept PDF or DOCX files, extract their text, split it into semantically meaningful chunks, generate vector embeddings, and persist everything in a local ChromaDB vector store — all running on an RTX 3060 6 GB GPU.

---

## Steps Performed

### Step 1 — Create the Project Directory Structure

The repository was organized into a clean, modular layout:

```
Project/
├── config.py               # Centralized configuration
├── init_vectordb.py         # Vector DB initialization script
├── requirements.txt         # Python dependencies
├── pipelines/
│   ├── __init__.py          # Makes pipelines a Python package
│   └── ingestion.py         # Full ingestion pipeline
└── data/
    ├── uploads/             # Stores copies of uploaded documents
    └── vectorstore/         # Persistent ChromaDB storage
```

**Why this structure?**

- `pipelines/` is a Python package so modules can be imported cleanly (e.g., `from pipelines.ingestion import ingest_document`).
- `data/` separates runtime artifacts (uploaded files, vector DB) from source code, keeping the repo clean.
- `config.py` lives at the root so every module can `import config` without path gymnastics.

---

### Step 2 — Define Dependencies (`requirements.txt`)

All third-party packages were declared in `requirements.txt`, grouped by purpose:

| Group | Packages | Purpose |
|-------|----------|---------|
| **Core Frameworks** | `langchain`, `langchain-community`, `langchain-huggingface`, `langserve`, `fastapi`, `uvicorn` | LangChain orchestration, API serving |
| **UI** | `gradio` | Web-based upload and chat interface (used in Phase 3) |
| **Vector Store** | `chromadb` | Local persistent vector database |
| **Embeddings** | `sentence-transformers` | Lightweight embedding models (GPU-accelerated) |
| **File Parsing** | `PyMuPDF`, `pdfplumber`, `python-docx` | Text extraction from PDF and DOCX |
| **LLM** | `llama-cpp-python` | Local quantized LLM inference via GGUF models (used in Phase 2) |
| **Utilities** | `python-dotenv`, `pydantic`, `pydantic-settings` | Environment variable loading, data validation |

A virtual environment was created and all packages were installed:

```bash
python3 -m venv venv
source venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
pip install -U langchain-chroma   # Updated Chroma integration
```

**VRAM consideration:** The embedding model (`all-MiniLM-L6-v2`) uses only ~80 MB of VRAM, leaving ample room for the LLM in later phases.

---

### Step 3 — Centralized Configuration (`config.py`)

All tunable parameters are collected in a single file so they can be adjusted without modifying pipeline code. Every setting can be overridden via environment variables.

#### Directory Paths

```python
BASE_DIR = Path(__file__).resolve().parent
UPLOAD_DIR = BASE_DIR / "data" / "uploads"
VECTORSTORE_DIR = BASE_DIR / "data" / "vectorstore"
```

- `BASE_DIR` resolves to the project root regardless of where Python is invoked.
- Both directories are auto-created on import (`mkdir(parents=True, exist_ok=True)`).

#### Chunking Settings

```python
CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", "1000"))
CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", "200"))
```

- **`CHUNK_SIZE = 1000`**: Each text chunk is at most 1000 characters. This is a balanced default — small enough for precise retrieval, large enough to retain context.
- **`CHUNK_OVERLAP = 200`**: Adjacent chunks share 200 characters to prevent information loss at boundaries.
- Both values are configurable per the project spec's requirement to "implement configurable chunk sizes to mitigate risks associated with large documents."

#### Embedding Model

```python
EMBEDDING_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
EMBEDDING_DEVICE = "cuda"
```

- **`all-MiniLM-L6-v2`**: A 22M-parameter model producing 384-dimensional embeddings. Chosen for its excellent speed-to-quality ratio and minimal VRAM footprint (~80 MB).
- **`cuda`**: Runs inference on the GPU for faster embedding generation.

#### Vector Store

```python
CHROMA_COLLECTION_NAME = "contracts"
```

- All document chunks are stored in a single Chroma collection named `contracts`.

#### LLM Settings (prepared for Phase 2)

```python
LLM_PROVIDER = "local"          # "local" or "openai"
LOCAL_MODEL_PATH = ""           # Path to a GGUF quantized model
LLM_N_CTX = 4096               # Context window size
LLM_N_GPU_LAYERS = 33          # Layers offloaded to GPU
LLM_TEMPERATURE = 0.1          # Low temperature for factual answers
LLM_MAX_TOKENS = 1024          # Max generation length
```

These are placeholders configured for Phase 2. The low temperature (0.1) is deliberate — contract Q&A requires factual, deterministic answers.

#### Server & File Support

```python
API_HOST = "0.0.0.0"
API_PORT = 8000
SUPPORTED_EXTENSIONS = {".pdf", ".docx"}
```

---

### Step 4 — Ingestion Pipeline (`pipelines/ingestion.py`)

This is the core module of Phase 1. It implements the full document ingestion flow in four stages:

```
Upload → Extract Text → Chunk → Embed & Store
```

#### 4.1 — File Parsing

Two dedicated extractors handle the supported file types:

**PDF Extraction** (`extract_text_from_pdf`):
```python
def extract_text_from_pdf(file_path: str | Path) -> str:
    text_parts: list[str] = []
    with fitz.open(str(file_path)) as doc:
        for page in doc:
            text_parts.append(page.get_text())
    return "\n".join(text_parts)
```
- Uses **PyMuPDF** (`fitz`) for fast, reliable PDF text extraction.
- Iterates through every page and concatenates the text with newlines.
- PyMuPDF was chosen over pdfplumber for its superior speed on large documents.

**DOCX Extraction** (`extract_text_from_docx`):
```python
def extract_text_from_docx(file_path: str | Path) -> str:
    doc = DocxDocument(str(file_path))
    return "\n".join(para.text for para in doc.paragraphs if para.text.strip())
```
- Uses **python-docx** to read paragraphs from Word documents.
- Filters out empty paragraphs to produce clean text.

**Router** (`extract_text`):
```python
def extract_text(file_path: str | Path) -> str:
    ext = Path(file_path).suffix.lower()
    if ext == ".pdf":
        return extract_text_from_pdf(path)
    elif ext == ".docx":
        return extract_text_from_docx(path)
    else:
        raise ValueError(...)
```
- Inspects the file extension and dispatches to the correct parser.
- Raises a clear error for unsupported file types.

#### 4.2 — Text Chunking

```python
def chunk_text(text, source, chunk_size=None, chunk_overlap=None) -> list[Document]:
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size or config.CHUNK_SIZE,
        chunk_overlap=chunk_overlap or config.CHUNK_OVERLAP,
        length_function=len,
        separators=["\n\n", "\n", ". ", " ", ""],
    )
    chunks = splitter.create_documents(
        texts=[text],
        metadatas=[{"source": source}],
    )
    for i, chunk in enumerate(chunks):
        chunk.metadata["chunk_index"] = i
    return chunks
```

**How `RecursiveCharacterTextSplitter` works:**

1. It tries to split on `"\n\n"` (paragraph breaks) first.
2. If chunks are still too large, it falls back to `"\n"` (line breaks).
3. Then `". "` (sentence boundaries), then `" "` (word boundaries).
4. As a last resort, it splits on individual characters.

This hierarchy preserves semantic coherence — paragraphs stay intact when possible, and sentences are only broken as a last resort.

**Metadata:** Each chunk carries:
- `source`: the original filename (for citation in answers).
- `chunk_index`: the position within the document (for ordering).

#### 4.3 — Embedding Model (Singleton)

```python
_embedding_model: HuggingFaceEmbeddings | None = None

def get_embedding_model() -> HuggingFaceEmbeddings:
    global _embedding_model
    if _embedding_model is None:
        _embedding_model = HuggingFaceEmbeddings(
            model_name=config.EMBEDDING_MODEL_NAME,
            model_kwargs={"device": config.EMBEDDING_DEVICE},
            encode_kwargs={"normalize_embeddings": True},
        )
    return _embedding_model
```

- **Singleton pattern**: The model is loaded once and reused across all calls, avoiding repeated GPU memory allocation.
- **`normalize_embeddings=True`**: Produces unit-length vectors so cosine similarity reduces to a dot product, improving retrieval accuracy and speed.
- **Device**: Loaded on CUDA (GPU) for fast batch encoding.

#### 4.4 — Vector Store Integration

```python
def get_vectorstore() -> Chroma:
    return Chroma(
        collection_name=config.CHROMA_COLLECTION_NAME,
        embedding_function=get_embedding_model(),
        persist_directory=str(config.VECTORSTORE_DIR),
    )

def add_documents_to_store(docs: list[Document]) -> Chroma:
    store = get_vectorstore()
    store.add_documents(docs)
    return store
```

- **ChromaDB** is configured with a persistent directory (`data/vectorstore/`), so embeddings survive process restarts.
- `add_documents` handles both embedding generation and storage in a single call.

#### 4.5 — Top-Level Entrypoint

```python
def ingest_document(file_path, chunk_size=None, chunk_overlap=None) -> dict:
```

This function orchestrates the full pipeline:

1. **Validate** the file extension against `SUPPORTED_EXTENSIONS`.
2. **Copy** the file to `data/uploads/` for archival.
3. **Extract** text using the appropriate parser.
4. **Validate** that the extracted text is non-empty.
5. **Chunk** the text with configurable parameters.
6. **Embed & store** all chunks in ChromaDB.
7. **Return** a statistics dictionary:

```python
{
    "filename": "contract.pdf",
    "characters": 3302,
    "chunks": 5,
    "chunk_size": 1000,
    "chunk_overlap": 200,
    "status": "success"
}
```

#### 4.6 — CLI Interface

The module can be run directly from the command line:

```bash
python -m pipelines.ingestion <file_path> [chunk_size] [chunk_overlap]
```

Examples:
```bash
# Default settings (chunk_size=1000, overlap=200)
python -m pipelines.ingestion contract.pdf

# Custom chunk settings
python -m pipelines.ingestion contract.pdf 500 100
```

---

### Step 5 — Vector DB Initialization Script (`init_vectordb.py`)

A standalone utility to create or reset the ChromaDB instance:

```python
def init_vectordb(reset: bool = False) -> None:
    if reset and config.VECTORSTORE_DIR.exists():
        shutil.rmtree(config.VECTORSTORE_DIR)
        config.VECTORSTORE_DIR.mkdir(parents=True, exist_ok=True)

    client = chromadb.PersistentClient(path=str(config.VECTORSTORE_DIR))
    collection = client.get_or_create_collection(
        name=config.CHROMA_COLLECTION_NAME,
    )
```

**Usage:**
```bash
# Create or verify the collection exists
python init_vectordb.py

# Wipe all data and start fresh
python init_vectordb.py --reset
```

- `--reset` deletes the entire `data/vectorstore/` directory and recreates it, useful when re-ingesting documents from scratch.
- Without `--reset`, it uses `get_or_create_collection` which is idempotent.

---

### Step 6 — Testing the Pipeline

The pipeline was tested end-to-end with the project's own spec document:

```bash
# 1. Initialize vector store
python init_vectordb.py --reset
# Output: Collection 'contracts' ready ─ 0 documents stored

# 2. Ingest a PDF
python -m pipelines.ingestion Smart_Contract_Assistant_Spec.pdf
# Output:
# Saved uploaded file to data/uploads/Smart_Contract_Assistant_Spec.pdf
# Split 'Smart_Contract_Assistant_Spec.pdf' into 5 chunks (size=1000, overlap=200)
# Loaded embedding model: sentence-transformers/all-MiniLM-L6-v2 on cuda
# Added 5 chunks to vector store 'contracts'
# Ingestion complete: {'filename': 'Smart_Contract_Assistant_Spec.pdf',
#                      'characters': 3302, 'chunks': 5,
#                      'chunk_size': 1000, 'chunk_overlap': 200,
#                      'status': 'success'}
```

**Results:**
- 3,302 characters extracted from a 4-page PDF.
- Split into 5 chunks of 1000 chars each with 200-char overlap.
- Embedded on GPU (CUDA) using `all-MiniLM-L6-v2`.
- Stored persistently in ChromaDB.
- Total ingestion time: ~13 seconds (includes first-time model download).

---

## Technology Choices Summary

| Component | Choice | Rationale |
|-----------|--------|-----------|
| PDF Parser | PyMuPDF | Fastest Python PDF library; handles complex layouts |
| DOCX Parser | python-docx | Standard library for Word documents |
| Text Splitter | LangChain `RecursiveCharacterTextSplitter` | Preserves semantic boundaries; configurable |
| Embedding Model | `all-MiniLM-L6-v2` | 22M params, 384 dims, ~80MB VRAM — ideal for 6GB GPU |
| Vector Store | ChromaDB (persistent) | Simple setup, built-in persistence, LangChain integration |
| Configuration | Environment variables + defaults | Flexible deployment without code changes |

---

## Phase 1 Deliverables Checklist

- [x] `requirements.txt` with all dependencies
- [x] Base folder structure (`pipelines/`, `data/uploads/`, `data/vectorstore/`)
- [x] `config.py` with centralized, overridable settings
- [x] `pipelines/ingestion.py` with extraction, configurable chunking, and embedding
- [x] `init_vectordb.py` for vector DB initialization and reset
- [x] Successful end-to-end test with a real PDF document
