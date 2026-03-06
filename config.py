import os
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent

# ── Directory Paths ──────────────────────────────────────────────
UPLOAD_DIR = BASE_DIR / "data" / "uploads"
VECTORSTORE_DIR = BASE_DIR / "data" / "vectorstore"

UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
VECTORSTORE_DIR.mkdir(parents=True, exist_ok=True)

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

# ── LLM Settings ────────────────────────────────────────────────
LLM_PROVIDER = os.getenv("LLM_PROVIDER", "local")  # "local" or "openai"
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")

# Local LLM (GGUF quantized model path for llama-cpp-python)
LOCAL_MODEL_PATH = os.getenv("LOCAL_MODEL_PATH", "")
LLM_N_CTX = int(os.getenv("LLM_N_CTX", "4096"))
LLM_N_GPU_LAYERS = int(os.getenv("LLM_N_GPU_LAYERS", "33"))
LLM_TEMPERATURE = float(os.getenv("LLM_TEMPERATURE", "0.1"))
LLM_MAX_TOKENS = int(os.getenv("LLM_MAX_TOKENS", "1024"))

# ── Server ───────────────────────────────────────────────────────
API_HOST = os.getenv("API_HOST", "0.0.0.0")
API_PORT = int(os.getenv("API_PORT", "8000"))

# ── Supported File Extensions ────────────────────────────────────
SUPPORTED_EXTENSIONS = {".pdf", ".docx"}
