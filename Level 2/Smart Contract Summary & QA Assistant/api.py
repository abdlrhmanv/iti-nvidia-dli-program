"""
FastAPI & LangServe Backend Entrypoint.

Exposes the RAG components (ingestion, retrieval, summarization) as REST APIs.
Also mounts the Gradio UI so both API and UI run on the same port.
"""

import logging
import os
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException
from fastapi.responses import RedirectResponse
import gradio as gr
from langserve import add_routes

import config
from app import build_ui
from pipelines.llm_pipeline import build_rag_chain, get_llm
from pipelines.vectorstore import get_vectorstore
# Used to expose raw LLM capabilities if needed
from langchain_core.prompts import ChatPromptTemplate

logging.basicConfig(
    level=logging.INFO,
    format="%(levelname)s | %(name)s | %(message)s",
)
logger = logging.getLogger(__name__)

config.ensure_dirs()


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Preload the LLM on startup
    logger.info("Preloading LLM for API...")
    try:
        get_llm()
        logger.info("LLM ready")
    except Exception as e:
        logger.warning("Could not preload LLM: %s", e)
    yield


app = FastAPI(
    title="Smart Contract Summary & QA API",
    version="1.0",
    description="A simple API server exposing RAG chains via LangServe.",
    lifespan=lifespan,
)


@app.get("/")
async def redirect_root_to_docs():
    """Redirect API root to the docs page, unless accessed via the Gradio mount."""
    return RedirectResponse("/docs")


# ── LangServe Routes ─────────────────────────────────────────────

from langchain_core.runnables import RunnablePassthrough
from pipelines.retrieval import retrieve_chunks, format_context

try:
    # Build complete end-to-end chain for LangServe
    # We create a Runnable that takes {"question": "..."}
    # fetches documents, formats them, and passes them to the LLM.
    
    def _fetch_and_format(inputs: dict) -> str:
        q = inputs["question"]
        docs = retrieve_chunks(q)
        return format_context(docs)

    # Note: RAG_PROMPT_TEMPLATE only takes {context} and {question}
    from pipelines.llm_pipeline import _rag_prompt
    
    setup_and_retrieval = RunnablePassthrough.assign(
        context=_fetch_and_format
    )
    
    full_rag_chain = setup_and_retrieval | _rag_prompt | get_llm()

    # 1. Provide a route to the RAG chain
    add_routes(
        app,
        full_rag_chain,
        path="/rag",
        disabled_endpoints=["playground"] # Custom context functions break standard playground UI sometimes
    )

except Exception as e:
    logger.error("Failed to add LangServe routes: %s", e)

# ── Mount Gradio UI ──────────────────────────────────────────────
# We mount the Gradio app at /ui so it coexists with the FastAPI routes

try:
    demo = build_ui()
    # Mounting at /ui because / is for API docs/redirect
    app = gr.mount_gradio_app(app, demo, path="/ui")
except Exception as e:
    logger.error("Failed to mount Gradio UI: %s", e)

if __name__ == "__main__":
    import uvicorn
    logger.info("Starting API server on %s:%d", config.API_HOST, config.API_PORT)
    uvicorn.run(app, host=config.API_HOST, port=config.API_PORT)
