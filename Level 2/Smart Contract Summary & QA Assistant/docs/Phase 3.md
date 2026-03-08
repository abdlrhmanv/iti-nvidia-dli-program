# Phase 3: Gradio UI Integration

## Overview

Phase 3 adds a web-based UI using Gradio so users can upload documents and ask questions without using the CLI. The app provides a **single chat window**: upload a document, optionally request a **full-document summary**, and ask questions with **streaming answers** and **source citations**. Each new upload clears the vector store so the chat is always about the current document.

---

## Steps Performed

### Step 1 — Frontend Entry Point (`app.py`)

`app.py` is the single entry point for the Gradio frontend. It:

1. Imports and uses the existing **ingestion**, **LLM**, **summarization**, and **vectorstore** pipelines (no duplication of logic).
2. Builds a **Gradio Blocks** interface with one unified view: file upload row, document status, and **chat** (conversation history + message input).
3. Renders answers and sources in Markdown; supports **multi-turn** follow-up questions with conversation context and **streaming** responses.

#### 1.1 — Upload & Document Actions

- **File input:** `gr.File` restricted to `.pdf` and `.docx` (matching `config.SUPPORTED_EXTENSIONS`).
- **"Load document":** Passes the file to `ingest_document()` from `pipelines.ingestion`. Before ingesting, `clear_vectorstore()` is called so the store contains only the new document. Status shows filename, chunk count, character count, and chunk size.
- **"Summarize":** Calls `summarize_document()` from `pipelines.summarization` and posts the result into the chat (full-document summary using the current vector store).
- **"Clear chat":** Clears the conversation history; the loaded document remains in the vector store.

**Flow:** User selects file → clicks "Load document" → store is cleared, file is copied to `data/uploads/`, extracted, chunked, embedded, stored → status updates; user can then summarize or ask questions.

#### 1.2 — Chat (Streaming, Multi-turn)

- **Input:** A textbox for the user's question; conversation history is kept in a `gr.Chatbot`.
- **Action:** On "Send" (or Enter), the question is passed to `stream_answer_question()` from `pipelines.llm_pipeline` with `chat_history` for follow-up awareness.
- **Output:** The answer is **streamed** token-by-token into the chat; when done, source references (filename, chunk index) are appended. Guardrails (citations, disclaimer) apply as in Phase 2.

**Flow:** User types a question → retrieval runs → LLM streams the answer → guardrails applied → answer and sources shown in the chat.

#### 1.3 — Server Configuration

- Host and port are read from `config.API_HOST` and `config.API_PORT` (defaults: `127.0.0.1`, `8000`).
- `config.ensure_dirs()` is called at import so `data/uploads/` and `data/vectorstore/` exist before Gradio handles files.
- The LLM is preloaded at startup so the first query does not stall.

---

## Design Choices

| Decision | Rationale |
|----------|-----------|
| Single `app.py` | Keeps UI logic in one place; can also run via `api.py` with Gradio mounted at `/ui`. |
| Single chat view (upload + chat together) | One document per session; upload clears store so context is always the current file. |
| Streaming answers | Better perceived latency; uses `stream_answer_question` with conversation history for follow-ups. |
| Summarize button | Optional full-document summary via `pipelines.summarization` (stuff or map_reduce by context size). |
| Markdown for answers and sources | Citations and formatting display correctly in the chatbot. |
| Reuse `ingest_document`, `stream_answer_question`, `summarize_document` | No duplication; all behavior stays in pipelines. |
| Errors surfaced in UI | Users see validation and pipeline errors in the chat. |

---

## How to Run

1. Ensure the environment is set up (Phase 1) and the vector store is initialized (e.g. run `python -m scripts.init_vectordb` if needed).
2. Configure the LLM (Phase 2): set `LOCAL_MODEL_PATH` in `.env` for local inference, or `OPENAI_API_KEY` for OpenAI.
3. From the project root:

   ```bash
   python app.py
   ```

4. Open the URL shown in the terminal (e.g. `http://127.0.0.1:8000`).
5. Use **Load document** to upload a PDF/DOCX (this clears the store and ingests the file), then use **Summarize** or type questions in the chat. Answers stream with source citations.

**Alternative:** Run `python api.py` for FastAPI + LangServe; open `/ui` for the Gradio app and `/docs` for the API.

---

## Phase 3 Deliverables Checklist

- [x] `app.py` as the frontend entry point
- [x] Single view: file upload connected to `pipelines.ingestion.ingest_document`; store cleared per document via `clear_vectorstore`
- [x] Chat: question input connected to `pipelines.llm_pipeline.stream_answer_question` with conversation history; streaming responses
- [x] Optional summarization via `pipelines.summarization.summarize_document` (Summarize button)
- [x] Answers displayed with source citations (Markdown in chat + source list in message)
- [x] Documentation (`Phase 3.md`)
