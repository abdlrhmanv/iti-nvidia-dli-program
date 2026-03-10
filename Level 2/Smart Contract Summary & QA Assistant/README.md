# Smart Contract Summary & QA Assistant

A RAG-powered web application enabling users to upload legal documents (contracts, insurance policies, reports) and interact with them via a conversational assistant. Built with **FastAPI**, **LangServe**, **LangChain**, and **Gradio**.

This project aligns with the NVIDIA DLI Course requirements (Level 2).

## 🚀 Features

- **File Ingestion:** Supports PDF and DOCX uploads with automatic text extraction (PyMuPDF / python-docx).
- **Parent-Child Chunking:** Documents are split into large parent chunks (for LLM context) linked to small child chunks (for precise embedding retrieval) via `MultiVectorRetriever`.
- **Vector Search:** Embeds child chunks using `sentence-transformers/all-MiniLM-L6-v2` into a persistent **ChromaDB** instance. Parent chunks are stored in a `LocalFileStore` docstore.
- **Conversational QA:** Multi-turn Q&A with streaming responses, inline `[Source N]` citations, grounding guardrails, legal disclaimers, and suggested follow-up questions. **Answer depth scales proportionally** with the amount of relevant context retrieved.
- **Document Summarization:** Dedicated pipeline using custom prompts with **proportional summary length** — small documents get concise summaries, large documents get detailed, structured summaries with section headings and bullet points. Uses `stuff` chain for small docs and `map_reduce` for large ones.
- **3-Tier LLM Fallback:** Automatically selects the best available LLM: Local GGUF model (Mistral-7B) → Groq API (llama-3.3-70b) → OpenAI (gpt-4o-mini).
- **REST APIs:** Full integration with LangServe, exposing the RAG chain at `/rag` for programmatic access.
- **Evaluation Pipeline:** LLM-as-a-Judge evaluation scoring Context Relevance and Answer Groundedness.

## 🛠 Prerequisites & Installation

### 1. Requirements
- Python 3.10+
- (Optional) CUDA toolkit for GPU acceleration with local models.

### 2. Install Dependencies
```bash
pip install -r requirements.txt
```

### 3. Environment Configuration
```bash
cp .env.example .env
```
_Key settings to configure:_
- **Local Models:** Set `LOCAL_MODEL_PATH` to a downloaded `.gguf` model file.
- **Cloud Models:** Set `GROQ_API_KEY` (recommended) or `OPENAI_API_KEY` as fallback.
- **Chunking:** `PARENT_CHUNK_SIZE` (default: 2000), `CHILD_CHUNK_SIZE` (default: 400).
- **Server:** `API_HOST` and `API_PORT` (default: `127.0.0.1:8000`).

---

## 🏃 Running the Application

### Start the Server (API + Gradio UI)
```bash
python api.py
```

### Accessing the Interfaces
- **Web UI:** `http://127.0.0.1:8000/ui` — Upload documents, ask questions, summarize.
- **API Docs:** `http://127.0.0.1:8000/docs` — Interactive Swagger documentation.
- **RAG API:** `http://127.0.0.1:8000/rag/invoke` — Programmatic RAG chain access.

---

## 📊 Evaluation

An automated evaluation pipeline tests answer quality using an LLM-as-a-Judge approach, scoring **Context Relevance** and **Answer Groundedness** (0–5 scale) across in-domain and out-of-domain questions.

```bash
python -m scripts.evaluate
```
Results are saved to `docs/Evaluation_Report.md`.

---

## 🗂 Project Structure

```
Smart Contract Summary & QA Assistant/
├── api.py              # FastAPI + LangServe + Gradio mount
├── app.py              # Gradio UI (Upload, Summarize, Q&A Chat)
├── config.py           # Centralized configuration (.env backed)
├── requirements.txt    # Python dependencies
├── models/             # Local GGUF model files
├── data/
│   ├── uploads/        # Persisted uploaded documents
│   ├── vectorstore/    # ChromaDB persistent store
│   └── docstore/       # Parent chunk file store
├── pipelines/
│   ├── ingestion.py    # Text extraction + Parent-Child chunking
│   ├── vectorstore.py  # Chroma + Docstore + MultiVectorRetriever
│   ├── retrieval.py    # Query → ranked parent documents
│   ├── llm_pipeline.py # Prompts, LLM, guardrails, streaming
│   └── summarization.py# Proportional map-reduce summarization
├── scripts/
│   └── evaluate.py     # LLM-as-a-Judge evaluation pipeline
└── docs/               # Specs, reports, walkthrough PDF
```

## 🔧 Key Design Decisions

| Decision | Rationale |
|---|---|
| **Parent-Child Chunking** | Small chunks for precise retrieval, large chunks for rich LLM context |
| **3-Tier LLM Fallback** | Works offline (local GGUF), fast (Groq), or anywhere (OpenAI) |
| **Proportional Summarization** | Summary length scales with document size (150–1500 words) |
| **Output Guardrails** | Post-generation citation checks + legal disclaimer |
| **Streaming Responses** | Token-by-token output for responsive UX |
| **Singleton Patterns** | Embedding model & LLM loaded once to avoid reload latency |
