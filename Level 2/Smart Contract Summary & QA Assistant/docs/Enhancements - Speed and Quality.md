# Enhancements: Speed & Human-like Response Quality

This document lists concrete changes to make the app **faster** and responses **more natural and human-like**, with where to implement them and trade-offs.

---

## Part 1 — Speed

### 1.1 Use Groq or OpenAI for Q&A (easiest, big win)

**Why:** Cloud LLMs are much faster than a local 7B model and often higher quality. Groq is preferred when `GROQ_API_KEY` is set.

**Groq model choice (best fit for contract Q&A):**
- **`llama-3.3-70b-versatile`** (default) — Best balance of quality and speed for document Q&A and summarization.
- **`llama-3.1-8b-instant`** — Fastest; good for quick answers.
- **`groq/compound-mini`** — Lightweight, fast.
- **`meta-llama/llama-4-scout-17b-16e-instruct`** — Newer Llama 4; good for instruction-following.

Set `GROQ_MODEL` in `.env` to switch. (Guard/safeguard and whisper models are for moderation/speech, not chat.)

**How:** Set `GROQ_API_KEY` in `.env` (default model: `llama-3.3-70b-versatile`) or `OPENAI_API_KEY`; leave `LOCAL_MODEL_PATH` empty to use cloud. The app prefers Groq → OpenAI when both are set.

**Where:** `config.py` (env), `pipelines/llm_pipeline.py` (already has fallback).

---

### 1.2 Lower max tokens for Q&A

**Why:** Most answers don’t need 2048 tokens. Capping at 512–768 speeds up generation.

**How:** Add `LLM_MAX_TOKENS_QA` in config (e.g. 512) and use it in the RAG chain; keep `LLM_MAX_TOKENS` for summarization.

**Where:** `config.py`, `pipelines/llm_pipeline.py` (pass `max_tokens` when building the chain for Q&A).

---

### 1.3 Fewer retrieval chunks when possible

**Why:** Fewer chunks → smaller context → faster LLM and sometimes more focused answers.

**How:** Reduce `RETRIEVAL_TOP_K` (e.g. 3–4) in config, or make it dynamic: 3 for short questions, 5 for “explain”/“summarize” (e.g. based on query length or keywords).

**Where:** `config.py`, `pipelines/retrieval.py`.

---

### 1.4 Faster summarization (already partly done)

**Done:** Merging chunks into super-chunks reduces map steps.

**Further options:**
- Use **OpenAI for summarization only** (one API call for “stuff” or few map_reduce steps).
- **Smaller/faster local model** for summarization (e.g. Phi-2, TinyLlama) and keep Mistral for Q&A.

**Where:** `pipelines/summarization.py`, `config.py` (optional `SUMMARIZATION_PROVIDER`).

---

### 1.5 Embedding and ingestion

**Current:** GPU embeddings, single-doc ingestion.

**Optional:**
- **Larger embedding batch** during ingestion (if the API supports it) to speed up big PDFs.
- **Lazy-load embedding model** on first use (already done); ensure it’s not loaded at import.

**Where:** `pipelines/ingestion.py`, `pipelines/vectorstore.py`.

---

## Part 2 — Better, more human-like responses

### 2.1 Prompt design (high impact)

**Why:** “Be concise and factual” and rigid “Cite [Source N]” lead to stiff, robotic answers.

**Improvements:**
- Ask for a **conversational, helpful tone**: short sentences, natural flow, no unnecessary bullet lists unless the user asks.
- **Citation style:** “Mention sources inline when you refer to a specific detail (e.g. ‘…as stated in the contract [Source 2].’) Do not start every sentence with ‘According to Source N’.”
- Add 1–2 **few-shot examples** of a natural answer with citations (optional).
- For **follow-ups:** “Use the conversation history only to resolve references like ‘it’, ‘that clause’, ‘the same amount’; keep answers grounded in the document excerpts.”

**Where:** `pipelines/llm_pipeline.py` — `RAG_PROMPT_TEMPLATE`, `RAG_PROMPT_WITH_HISTORY`.

---

### 2.2 Slightly higher temperature for Q&A

**Why:** Temperature 0.1 is very deterministic; 0.25–0.35 allows more natural variation while staying factual.

**How:** Use a separate `LLM_TEMPERATURE_QA` (e.g. 0.3) for the RAG chain and keep a lower value for summarization if desired.

**Where:** `config.py`, `pipelines/llm_pipeline.py` (use different temperature when creating the LLM for Q&A vs summarization, or one config that’s “good for both”).

---

### 2.3 Better retrieval quality

**Why:** Better chunks → better context → more relevant, natural answers.

**Options:**
- **Reranking:** Retrieve top_k=10, then rerank with a small cross-encoder or Cohere rerank → top 3–4. Better precision.
- **Hybrid search:** Combine semantic (current) + keyword (BM25) and merge/dedupe. Helps with names, dates, exact phrases.
- **Better chunking:** Semantic boundaries (e.g. split on headings/paragraphs first), or slightly larger chunks (1200–1500) with overlap so each chunk is more self-contained.

**Where:** `pipelines/retrieval.py`, `pipelines/ingestion.py` (chunking), new `pipelines/rerank.py` if needed.

---

### 2.4 Stronger embedding model (quality vs speed)

**Why:** all-MiniLM-L6-v2 is fast but small; larger models (e.g. all-mpnet-base-v2, BAAI/bge-small-en-v1.5) often improve retrieval quality.

**Trade-off:** Slightly more VRAM and latency; better relevance.

**Where:** `config.py` (`EMBEDDING_MODEL`), `pipelines/vectorstore.py` (uses config).

---

### 2.5 Post-processing / guardrails

**Current:** Citation check, legal disclaimer.

**Improvements:**
- **Soften disclaimer:** e.g. “This is an AI-generated summary for convenience, not legal advice.”
- **Don’t force citations** into every answer; only when the model already used them or when the answer is long. Avoid appending raw “Sources: 1, 2, 3” if the model already weaved them in.

**Where:** `pipelines/llm_pipeline.py` (`apply_output_guardrails`), `app.py` (how sources are appended to the message).

---

## Part 3 — Quick wins (implemented or easy to add)

| Enhancement | Impact | Effort | Where |
|-------------|--------|--------|--------|
| Human-like RAG prompts | High | Low | `llm_pipeline.py` |
| Temperature 0.25–0.35 for Q&A | Medium | Low | `config.py`, `llm_pipeline.py` |
| Prefer OpenAI when key set | Speed: High | None (already supported) | `.env` |
| Lower max_tokens for Q&A (512) | Speed: Medium | Low | `config.py`, `llm_pipeline.py` |
| RETRIEVAL_TOP_K = 4 | Speed: Low, focus: Medium | Low | `config.py` |
| Reranker (e.g. cross-encoder) | Quality: High | Medium | New module + `retrieval.py` |
| Better embedding model | Quality: Medium | Low | `config.py` |

---

## Summary

- **Speed:** Prefer OpenAI when possible; lower Q&A max_tokens; keep merged-chunk summarization; optionally reduce top_k.
- **Human-like:** Rewrite RAG prompts for tone and natural citations; raise Q&A temperature slightly; improve retrieval (rerank/hybrid/chunking) and optionally use a stronger embedding model.

Implementing the prompt and temperature changes gives the largest perceived improvement for the least code change; the rest can be added incrementally.
