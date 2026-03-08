# Phase 4: Additional Features

## Overview

After Phases 1–3, the project was extended with: a **FastAPI + LangServe** backend exposing the RAG chain as an API with the Gradio UI mounted at `/ui`; a **full-document summarization** pipeline; and an **evaluation script** that runs test questions and uses the LLM as a judge to score context relevance and answer groundedness.

---

## 1. API Backend (`api.py`)

### Purpose

- Expose the RAG pipeline as a REST API for integration with other tools or frontends.
- Serve the same Gradio UI from the same process so one server provides both API and UI.

### What It Does

- **FastAPI** app with lifespan hook that preloads the LLM on startup.
- **LangServe** route at `/rag`: accepts `{"question": "..."}` and returns the RAG answer. The chain is built from `retrieve_chunks` → `format_context` → `_rag_prompt` → LLM.
- **Gradio** UI mounted at `/ui` (root `/` redirects to `/docs` for API documentation).
- Host and port from `config.API_HOST` and `config.API_PORT`.

### How to Run

```bash
python api.py
# Open http://127.0.0.1:8000/docs for API docs
# Open http://127.0.0.1:8000/ui for the Gradio app
```

### Dependencies

Already in `requirements.txt`: `fastapi`, `uvicorn`, `langserve`, `sse_starlette`.

---

## 2. Summarization Pipeline (`pipelines/summarization.py`)

### Purpose

Generate a full-document summary of the currently loaded document (all chunks in the vector store), either for quick overview or to answer “summarize this document” in the UI.

### What It Does

- Reads all documents from the Chroma collection (via `get_vectorstore().get()`), reconstructs LangChain `Document` objects, and sorts by `chunk_index`.
- Chooses **chain type** by estimated token count:
  - **stuff**: when content fits in context (within `LLM_N_CTX - LLM_MAX_TOKENS - 500`), single prompt with all chunks.
  - **map_reduce**: when content is larger, to avoid context overflow.
- Uses `load_summarize_chain` from `langchain_classic` (pulled in by `langchain-community`) with the same LLM as the RAG pipeline (`get_llm()`).
- Returns a string summary or an error message if the store is empty or the chain fails.

### Usage

- **In the UI:** User clicks “Summarize”; the result is posted into the chat.
- **CLI:** `python -m pipelines.summarization` (requires a document already ingested).

---

## 3. Evaluation Script (`scripts/evaluate.py`)

### Purpose

Assess RAG quality on a fixed set of test questions using an **LLM-as-judge** to score **context relevance** and **answer groundedness**, and write a markdown report.

### What It Does

- Expects a sample document at `config.UPLOAD_DIR / "vertrag-ueber-freie-mitarbeiter-englisch-data-data.pdf"` (or places any test PDF there and ingests it).
- For each question in `TEST_QA_PAIRS`:
  1. Calls `answer_question(question)` to get the RAG answer.
  2. Retrieves the same top-k chunks used for context.
  3. Invokes a judge prompt (with question, context, and answer) and parses JSON for `context_relevance` (0–5), `answer_groundedness` (0–5), and `reasoning`.
- Aggregates average relevance and groundedness, measures latency per query.
- Writes **`docs/Evaluation_Report.md`** with date, model name, overall metrics, and per-question details (answer, latency, sources count, scores, reasoning).

### How to Run

```bash
# From project root; ensure a test PDF is in data/uploads/ or the script will ingest the default path if present
python -m scripts.evaluate
```

### Limitations (noted in the report)

- Context relevance depends on the embedding model (e.g. MiniLM).
- LLM-as-judge (e.g. 7B local) may occasionally output invalid JSON.
- Latency is tied to local GPU and layer offloading.

---

## 4. Advanced RAG Enhancements & UI Polish

### Purpose

Improve the quality of LLM answers by providing richer context without exceeding token limits, and enhance the user experience with transparency and interactivity.

### What It Does

- **Parent-Child Chunking (Small-to-Big Retrieval):** 
  - Splits documents into large `Parent` chunks (2000 chars) for LLM context, and small `Child` chunks (400 chars) for ChromaDB semantic search.
  - Links them using `MultiVectorRetriever` and an `EncoderBackedStore` over a `LocalFileStore` (`data/docstore`).
  - Ensures the LLM receives the full context of legal clauses while maintaining high search accuracy.
- **Enhanced Gradio UI (`app.py`):**
  - **View Retrieved Context Accordion:** A togglable `<details>` block that lets users inspect the raw Parent Chunk text retrieved from the vector database.
  - **Suggested Follow-up Questions:** The LLM prompt now automatically generates 3 contextual follow-up questions at the end of each answer.
  - **Sample Documents:** Users can quickly test the app using pre-loaded documents (`gr.Examples`).
  - **Prompt Guardrails:** Strict prompt instructions enforce bulleted lists, precise `[Source N]` inline citations, and fallback behaviors for out-of-scope questions.

---

## Deliverables Checklist

- [x] `api.py`: FastAPI + LangServe RAG route and Gradio at `/ui`
- [x] `pipelines/summarization.py`: full-document summary (stuff / map_reduce)
- [x] `scripts/evaluate.py`: LLM-as-judge evaluation and `docs/Evaluation_Report.md`
- [x] `pipelines/vectorstore.py` & `ingestion.py`: Parent-Child Retrieval Architecture
- [x] `app.py`: Upgraded UI with Expandable Context, Follow-ups, and Examples
- [x] Documentation (this file)
