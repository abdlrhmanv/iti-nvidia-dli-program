import os
from pathlib import Path

from dotenv import load_dotenv

BASE_DIR = Path(__file__).resolve().parent

# Load .env from project root so LOCAL_MODEL_PATH, etc. are set
load_dotenv(BASE_DIR / ".env")

# ── Core Directory Paths ──────────────────────────────────────────
DATA_DIR = BASE_DIR / "data"
UPLOAD_DIR = DATA_DIR / "uploads"
VECTORSTORE_DIR = DATA_DIR / "vectorstore"
DOCSTORE_DIR = DATA_DIR / "docstore"  # LocalFileStore for Parent-Child
DOCS_DIR = BASE_DIR / "docs"
SCRIPTS_DIR = BASE_DIR / "scripts"

# ── Chunking Settings (Parent-Child Strategy) ───────────────────
# Parent chunks map to larger logical sections (passed to LLM)
PARENT_CHUNK_SIZE = int(os.getenv("PARENT_CHUNK_SIZE", "2000"))
PARENT_CHUNK_OVERLAP = int(os.getenv("PARENT_CHUNK_OVERLAP", "400"))

# Child chunks map to smaller semantic concepts (embedded in Chroma)
CHILD_CHUNK_SIZE = int(os.getenv("CHILD_CHUNK_SIZE", "400"))
CHILD_CHUNK_OVERLAP = int(os.getenv("CHILD_CHUNK_OVERLAP", "50"))

# ── Embedding Model (small footprint for RTX 3060 6GB) ──────────
EMBEDDING_MODEL_NAME = os.getenv(
    "EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2"
)
EMBEDDING_DEVICE = os.getenv("EMBEDDING_DEVICE", "cuda")

# ── Vector Store ─────────────────────────────────────────────────
CHROMA_COLLECTION_NAME = os.getenv("CHROMA_COLLECTION", "contracts")

# ── Retrieval ────────────────────────────────────────────────────
RETRIEVAL_TOP_K = int(os.getenv("RETRIEVAL_TOP_K", "5"))

# ── LLM Settings (local GGUF model via llama-cpp-python) ────────
_raw_local_path = os.getenv("LOCAL_MODEL_PATH", "").strip()
# Resolve relative paths against project root so .env works from any cwd
if _raw_local_path and not Path(_raw_local_path).is_absolute():
    LOCAL_MODEL_PATH = str((BASE_DIR / _raw_local_path).resolve())
else:
    LOCAL_MODEL_PATH = _raw_local_path

LLM_N_CTX = int(os.getenv("LLM_N_CTX", "4096"))
LLM_N_GPU_LAYERS = int(os.getenv("LLM_N_GPU_LAYERS", "33"))
# 0.25–0.35 gives more natural, varied answers; 0.1 is more deterministic (good for summarization)
LLM_TEMPERATURE = float(os.getenv("LLM_TEMPERATURE", "0.25"))
LLM_MAX_TOKENS = int(os.getenv("LLM_MAX_TOKENS", "2048"))
# Optional: cap Q&A length for faster replies (default: use LLM_MAX_TOKENS)
LLM_MAX_TOKENS_QA = int(os.getenv("LLM_MAX_TOKENS_QA", "0"))  # 0 = use LLM_MAX_TOKENS

# ── Groq (fast cloud LLM; preferred over OpenAI when key is set) ─
# Best for contract Q&A/summarization: llama-3.3-70b-versatile (quality).
# Faster option: llama-3.1-8b-instant or groq/compound-mini.
GROQ_API_KEY = os.getenv("GROQ_API_KEY", "").strip()
GROQ_MODEL = os.getenv("GROQ_MODEL", "llama-3.3-70b-versatile")

# ── OpenAI (fallback when no local model and no Groq key) ─
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "").strip()
OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-4o-mini")

# ── Server ───────────────────────────────────────────────────────
API_HOST = os.getenv("API_HOST", "127.0.0.1")
API_PORT = int(os.getenv("API_PORT", "8000"))

# ── Supported File Extensions ────────────────────────────────────
SUPPORTED_EXTENSIONS = {".pdf", ".docx"}


def ensure_dirs() -> None:
    """Create runtime directories if they don't exist."""
    for d in (UPLOAD_DIR, VECTORSTORE_DIR):
        d.mkdir(parents=True, exist_ok=True)


# Auto-create on first import so the rest of the app can assume they exist.
ensure_dirs()
