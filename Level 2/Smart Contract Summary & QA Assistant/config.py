import os
from pathlib import Path

from dotenv import load_dotenv

BASE_DIR = Path(__file__).resolve().parent

# Load .env from project root so LOCAL_MODEL_PATH, etc. are set
load_dotenv(BASE_DIR / ".env")

# ── Directory Paths ──────────────────────────────────────────────
DATA_DIR = BASE_DIR / "data"
UPLOAD_DIR = DATA_DIR / "uploads"
VECTORSTORE_DIR = DATA_DIR / "vectorstore"
DOCS_DIR = BASE_DIR / "docs"
SCRIPTS_DIR = BASE_DIR / "scripts"

# ── Chunking Settings ────────────────────────────────────────────
CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", "1000"))
CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", "200"))

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
LLM_TEMPERATURE = float(os.getenv("LLM_TEMPERATURE", "0.1"))
LLM_MAX_TOKENS = int(os.getenv("LLM_MAX_TOKENS", "2048"))

# ── OpenAI (fallback when LOCAL_MODEL_PATH not set or file missing) ─
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
