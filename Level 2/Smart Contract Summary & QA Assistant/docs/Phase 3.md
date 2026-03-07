# Phase 3: Gradio UI Integration

## Overview

Phase 3 adds a web-based UI using Gradio so users can upload documents and ask questions without using the CLI. The app provides dedicated tabs for **file upload** (ingestion) and **chat** (RAG Q&A), and displays answers with **source citations** and a relevance-ranked source list.

---

## Steps Performed

### Step 1 — Frontend Entry Point (`app.py`)

`app.py` is the single entry point for the Gradio frontend. It:

1. Imports and uses the existing **ingestion** and **LLM** pipelines (no duplication of logic).
2. Builds a **Gradio Blocks** interface with two tabs: **Upload** and **Chat**.
3. Renders answers and sources in Markdown for clear citation display.

#### 1.1 — Upload Tab

- **File input:** `gr.File` restricted to `.pdf` and `.docx` (matching `config.SUPPORTED_EXTENSIONS`).
- **Action:** On "Ingest document", the file is passed to `ingest_document()` from `pipelines.ingestion`.
- **Output:** A Markdown status message with filename, character count, number of chunks, and chunk size/overlap. Errors (e.g. unsupported type, empty document) are shown in the same area.

**Flow:** User selects file → clicks "Ingest document" → file is copied to `data/uploads/`, extracted, chunked, embedded, and stored in ChromaDB → status message confirms success or reports an error.

#### 1.2 — Chat Tab

- **Input:** A textbox for the user's question.
- **Action:** On "Get answer", the question is passed to `answer_question()` from `pipelines.llm_pipeline`.
- **Outputs:**
  - **Answer:** The guardrailed answer (with `[Source N]` citations and disclaimer) rendered as Markdown.
  - **Sources:** A list of sources with filename, chunk index, and relevance score.

**Flow:** User types a question → clicks "Get answer" → retrieval runs → LLM generates answer → guardrails applied → answer and source list are displayed.

#### 1.3 — Server Configuration

- Host and port are read from `config.API_HOST` and `config.API_PORT` (defaults: `0.0.0.0`, `8000`), consistent with the rest of the project.
- `config.ensure_dirs()` is called at import so `data/uploads/` and `data/vectorstore/` exist before Gradio handles files.

---

## Design Choices

| Decision | Rationale |
|----------|-----------|
| Single `app.py` | Matches Phase 3 expected output; keeps UI logic in one place. |
| Tabs for Upload vs Chat | Clear separation of “add documents” and “ask questions” as in the PRD. |
| Markdown for answers and sources | Citations and formatting (e.g. bold, lists) display correctly. |
| Reuse `ingest_document` and `answer_question` | No duplication; ingestion and RAG behavior stay in pipelines. |
| Errors surfaced in UI | Users see validation and pipeline errors without using the CLI. |

---

## How to Run

1. Ensure the environment is set up (Phase 1) and the vector store is initialized (e.g. at least one document ingested, or run `scripts/init_vectordb.py` if needed).
2. Configure the LLM (Phase 2): set `LOCAL_MODEL_PATH` in `.env` for local inference.
3. From the project root:

   ```bash
   python app.py
   ```

4. Open the URL shown in the terminal (e.g. `http://127.0.0.1:8000`).
5. Use **Upload** to add PDF/DOCX documents, then **Chat** to ask questions and view answers with citations.

---

## Phase 3 Deliverables Checklist

- [x] `app.py` as the frontend entry point
- [x] Upload tab: file upload connected to `pipelines.ingestion.ingest_document`
- [x] Chat tab: question input connected to `pipelines.llm_pipeline.answer_question`
- [x] Answers displayed with source citations (Markdown + sources list)
- [x] Documentation (`Phase 3.md`)
