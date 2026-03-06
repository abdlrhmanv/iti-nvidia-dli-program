# Phase 2: Retrieval & LLM Answer Pipeline

## Overview

Phase 2 builds the core intelligence of the RAG system: a retrieval module that queries the ChromaDB vector store for semantically relevant chunks, and an LLM pipeline that generates grounded answers with citations and guardrails. By the end of this phase, the system can accept a natural language question, find the most relevant document chunks, and produce a factual answer with source references — all running locally on an RTX 3060 6 GB GPU.

---

## Steps Performed

### Step 1 — Retrieval Module (`pipelines/retrieval.py`)

This module provides the bridge between a user's question and the stored document embeddings.

#### 1.1 — Semantic Search (`retrieve_chunks`)

```python
def retrieve_chunks(query: str, top_k: Optional[int] = None) -> list[Document]:
    top_k = top_k or config.RETRIEVAL_TOP_K
    store = get_vectorstore()
    results = store.similarity_search_with_relevance_scores(query, k=top_k)
    ...
```

**How it works:**

1. The user's query is embedded using the same `all-MiniLM-L6-v2` model used during ingestion (from Phase 1).
2. ChromaDB computes cosine similarity between the query embedding and all stored chunk embeddings.
3. The top-k most similar chunks are returned, each with a relevance score.
4. Scores are attached to the document metadata for downstream use (e.g., filtering, display).

**Why `similarity_search_with_relevance_scores`?**

Unlike plain `similarity_search`, this method returns normalized relevance scores (0–1 for cosine), which enables:
- Quality thresholds (e.g., drop chunks below 0.3 relevance)
- Transparency in the UI (show users how confident the retrieval is)

**Default `top_k = 5`:** Balances recall (finding all relevant information) with precision (not overwhelming the LLM context window). Configurable via `RETRIEVAL_TOP_K` in config or environment.

#### 1.2 — Context Formatting (`format_context`)

```python
def format_context(docs: list[Document]) -> str:
    parts = []
    for i, doc in enumerate(docs, 1):
        source = doc.metadata.get("source", "unknown")
        chunk_idx = doc.metadata.get("chunk_index", "?")
        parts.append(f"[Source {i}: {source}, chunk {chunk_idx}]\n{doc.page_content}")
    return "\n\n---\n\n".join(parts)
```

Each chunk is tagged with a `[Source N]` label that the LLM is instructed to reference in its answer. This creates a clean citation trail from answer back to document.

**Example output:**
```
[Source 1: contract.pdf, chunk 2]
The contractor shall deliver all materials within 30 business days...

---

[Source 2: contract.pdf, chunk 5]
Payment terms: Net 60 from date of invoice...
```

#### 1.3 — CLI Interface

```bash
# Retrieve top-5 chunks (default)
python -m pipelines.retrieval "What are the payment terms?"

# Retrieve top-3 chunks
python -m pipelines.retrieval "What are the payment terms?" 3
```

---

### Step 2 — LLM Answer Pipeline (`pipelines/llm_pipeline.py`)

This module takes the retrieved context and generates a grounded, cited answer.

#### 2.1 — Prompt Engineering

The system prompt enforces strict grounding rules:

```
You are a precise contract analysis assistant. Your role is to answer
questions about uploaded documents using ONLY the provided context.

RULES YOU MUST FOLLOW:
1. Answer ONLY from the context provided below...
2. NEVER invent, assume, or hallucinate information...
3. Cite your sources using [Source N] tags...
4. Keep answers concise and factual.
5. If the question is ambiguous, ask for clarification...
6. For legal/contractual content, include a disclaimer...
```

**Why these rules matter:**

- **Rule 1–2** enforce grounding — the LLM must derive answers from the retrieved chunks, not from its parametric knowledge. This is critical for contract analysis where accuracy trumps creativity.
- **Rule 3** enables citations — the `[Source N]` tags map directly to the context chunks, allowing the UI to show which part of the document an answer came from.
- **Rule 6** adds legal safety — contract Q&A could be mistaken for legal advice.

The prompt uses LangChain's `ChatPromptTemplate` with a system/human message pair, compatible with both chat-based (OpenAI) and completion-based (llama-cpp) models.

#### 2.2 — Dual LLM Provider Support

The pipeline supports two providers, configured via `config.LLM_PROVIDER`:

| Provider | Implementation | Use Case |
|----------|---------------|----------|
| `"local"` | `LlamaCpp` from `langchain-community` | Default. Runs a GGUF quantized model on the local GPU. |
| `"openai"` | `ChatOpenAI` from `langchain-openai` | Fallback for higher-quality answers or when no local model is available. |

**Local LLM configuration (optimized for RTX 3060 6GB):**

```python
LlamaCpp(
    model_path=config.LOCAL_MODEL_PATH,     # Path to .gguf file
    n_ctx=config.LLM_N_CTX,                 # 4096 token context window
    n_gpu_layers=config.LLM_N_GPU_LAYERS,   # 33 layers on GPU
    temperature=config.LLM_TEMPERATURE,      # 0.1 (factual, deterministic)
    max_tokens=config.LLM_MAX_TOKENS,        # 1024 max generation
)
```

**VRAM budget:**
- Embedding model (`all-MiniLM-L6-v2`): ~80 MB
- LLM (e.g., Mistral-7B-Q4_K_M): ~4.5 GB
- Total: ~4.6 GB / 6 GB available — leaves headroom for ChromaDB and OS.

**Recommended local models for 6GB VRAM:**

| Model | Quantization | Size | Quality |
|-------|-------------|------|---------|
| Mistral-7B-Instruct | Q4_K_M | ~4.4 GB | Good general-purpose |
| Phi-3-mini-4k-instruct | Q4_K_M | ~2.3 GB | Fast, good for short context |
| Qwen2.5-7B-Instruct | Q4_K_M | ~4.7 GB | Strong reasoning |

The LLM is loaded as a singleton to avoid reloading the model on every query.

#### 2.3 — RAG Chain (LangChain LCEL)

The chain is composed using LangChain Expression Language:

```python
chain = prompt | llm | StrOutputParser()
```

This pipeline:
1. Formats the prompt with context + question.
2. Sends it to the LLM.
3. Parses the string output.

#### 2.4 — Output Guardrails

Post-generation guardrails enforce quality and safety:

```python
def apply_output_guardrails(answer: str, context_docs: list[Document]) -> str:
```

**Guardrail 1 — Empty/Short Answer Detection:**
If the LLM returns fewer than 10 characters, a fallback message is returned instead of showing garbage output.

**Guardrail 2 — Citation Check:**
If the answer doesn't contain any `[Source N]` citations but context was provided, the system appends a note listing which documents were used. This ensures traceability even if the LLM forgot to cite.

**Guardrail 3 — Legal Disclaimer:**
Every answer is appended with a disclaimer that this is AI-generated and not legal advice. Required by the PRD for contract-related content.

#### 2.5 — Public Entry Point (`answer_question`)

```python
def answer_question(question: str, top_k: Optional[int] = None) -> dict:
```

Orchestrates the full pipeline:

1. **Retrieve** relevant chunks from ChromaDB.
2. **Format** them into a numbered context block.
3. **Generate** an answer via the LangChain RAG chain.
4. **Apply** output guardrails (citation check, disclaimer).
5. **Return** a structured result:

```python
{
    "answer": "Based on [Source 1], the payment terms are...\n\n---\nDisclaimer: ...",
    "sources": [
        {"source": "contract.pdf", "chunk_index": 2, "relevance_score": 0.8721},
        {"source": "contract.pdf", "chunk_index": 5, "relevance_score": 0.8134},
    ],
    "raw_answer": "Based on [Source 1], the payment terms are..."
}
```

The `raw_answer` field is preserved for evaluation (Phase 5) where we may want to compare pre- and post-guardrail outputs.

#### 2.6 — CLI Interface

```bash
# Ask a question (uses default top_k=5)
python -m pipelines.llm_pipeline "What are the key obligations?"

# Ask with custom top_k
python -m pipelines.llm_pipeline "What are the payment terms?" 3
```

---

### Step 3 — Dependency Update

Added `langchain-openai>=0.2.0` to `requirements.txt` for the OpenAI fallback provider.

Install the new dependency:
```bash
pip install langchain-openai>=0.2.0
```

---

## Updated Project Structure

```
Smart Contract Summary & QA Assistant/
├── config.py                   # Centralized configuration
├── requirements.txt            # Python dependencies (updated)
├── .env.example                # Environment variable template
├── pipelines/
│   ├── __init__.py             # Public API re-exports
│   ├── ingestion.py            # Phase 1: extraction, chunking, embedding
│   ├── vectorstore.py          # Shared embedding model & ChromaDB access
│   ├── retrieval.py            # Phase 2: semantic search (NEW)
│   └── llm_pipeline.py         # Phase 2: LLM inference + guardrails (NEW)
├── scripts/
│   └── init_vectordb.py        # Vector DB initialization & reset
├── docs/                       # Project documentation & specs
└── data/                       # Auto-created at runtime
    ├── uploads/                # Stored document copies
    └── vectorstore/            # Persistent ChromaDB storage
```

---

## Setup Instructions for Local LLM

To use the local LLM provider, download a GGUF quantized model:

```bash
# Example: Download Mistral-7B-Instruct Q4_K_M
pip install huggingface-hub
huggingface-cli download TheBloke/Mistral-7B-Instruct-v0.2-GGUF \
    mistral-7b-instruct-v0.2.Q4_K_M.gguf \
    --local-dir ./models
```

Then set the path in your environment or `.env` file:
```bash
export LOCAL_MODEL_PATH="./models/mistral-7b-instruct-v0.2.Q4_K_M.gguf"
```

Or for a lighter model that uses less VRAM:
```bash
huggingface-cli download microsoft/Phi-3-mini-4k-instruct-gguf \
    Phi-3-mini-4k-instruct-q4.gguf \
    --local-dir ./models
export LOCAL_MODEL_PATH="./models/Phi-3-mini-4k-instruct-q4.gguf"
```

---

## Architectural Decisions

| Decision | Choice | Rationale |
|----------|--------|-----------|
| Retrieval method | Cosine similarity with scores | Enables quality thresholds and transparency |
| Context formatting | Numbered `[Source N]` tags | Creates a citation trail the LLM can reference |
| LLM loading | Singleton pattern | Avoids reloading 4+ GB model on every request |
| Guardrails | Post-generation (output filtering) | Simpler than constrained decoding; catches missing citations and adds disclaimers |
| Dual provider | Local (default) + OpenAI (fallback) | Flexibility for different environments and quality needs |
| Temperature | 0.1 | Contract Q&A demands factual, deterministic answers |

---

## Phase 2 Deliverables Checklist

- [x] `pipelines/retrieval.py` with semantic search and context formatting
- [x] `pipelines/llm_pipeline.py` with prompt engineering, dual LLM support, guardrails, and citations
- [x] Updated `requirements.txt` with `langchain-openai`
- [x] CLI interfaces for both modules
- [x] Documentation (`Phase 2.md`)
