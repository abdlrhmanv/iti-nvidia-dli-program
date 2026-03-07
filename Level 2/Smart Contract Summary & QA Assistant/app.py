"""
Gradio frontend: single chat window with document upload in the same view.
Upload a PDF/DOCX, then continue the conversation about that document.
Supports multi-turn follow-up questions with conversation context.
"""

from __future__ import annotations

import logging
from pathlib import Path

import gradio as gr

import config
from pipelines.ingestion import ingest_document
from pipelines.llm_pipeline import stream_answer_question, get_llm
from pipelines.vectorstore import clear_vectorstore

logging.basicConfig(
    level=logging.INFO,
    format="%(levelname)s | %(name)s | %(message)s",
)
logger = logging.getLogger(__name__)

config.ensure_dirs()


def _path_from_file(file) -> Path | None:
    if file is None:
        return None
    if isinstance(file, (list, tuple)) and file:
        file = file[0]
    path = Path(file.name) if hasattr(file, "name") else Path(file)
    return path if path.exists() else None


def build_ui():
    with gr.Blocks(title="Smart Contract Summary & QA Assistant") as demo:
        gr.Markdown(
            "# 📄 Smart Contract Summary & QA Assistant\n"
            "Upload a **PDF** or **DOCX** document, then ask questions about it. "
            "Follow-up questions are supported — the assistant remembers the conversation."
        )

        # ── Upload Row ──────────────────────────────────────────
        with gr.Row():
            with gr.Column(scale=3):
                file_input = gr.File(
                    label="Document (PDF or DOCX)",
                    file_types=[".pdf", ".docx"],
                    type="filepath",
                )
            with gr.Column(scale=1, min_width=160):
                upload_btn = gr.Button("📤 Load document", variant="primary", size="lg")
                clear_btn = gr.Button("🗑️ Clear chat", variant="secondary", size="sm")

        # ── Document status ─────────────────────────────────────
        doc_status = gr.Markdown(
            value="*No document loaded yet. Upload a PDF or DOCX to get started.*",
            elem_id="doc-status",
        )

        # ── Chat ────────────────────────────────────────────────
        chatbot = gr.Chatbot(
            label="Chat",
            height=450,
        )
        with gr.Row():
            msg_input = gr.Textbox(
                placeholder="Ask a question about the document...",
                label="Message",
                lines=1,
                show_label=False,
                scale=5,
            )
            send_btn = gr.Button("Send", variant="primary", scale=1)

        # ── Upload handler ──────────────────────────────────────
        def on_upload(file, history):
            path = _path_from_file(file)
            if path is None:
                return history, gr.File(value=None), "*No document loaded.*"
            ext = path.suffix.lower()
            if ext not in config.SUPPORTED_EXTENSIONS:
                return (
                    history + [{"role": "assistant", "content": f"❌ Unsupported file type: **{ext}**. Please use PDF or DOCX."}],
                    gr.File(value=None),
                    "*No document loaded.*",
                )
            try:
                clear_vectorstore()
                result = ingest_document(path)
                status = (
                    f"📄 **{result['filename']}** — "
                    f"{result['chunks']} chunks | "
                    f"{result['characters']:,} characters | "
                    f"chunk size: {result['chunk_size']}"
                )
                msg = (
                    f"✅ I've loaded **{result['filename']}** "
                    f"({result['chunks']} chunks, {result['characters']:,} characters). "
                    "Ask me anything about this document!"
                )
                return (
                    history + [{"role": "assistant", "content": msg}],
                    gr.File(value=None),
                    status,
                )
            except Exception as e:
                logger.exception("Upload failed")
                return (
                    history + [{"role": "assistant", "content": f"**Error:** {e}"}],
                    gr.File(value=None),
                    "*Upload failed.*",
                )

        upload_btn.click(
            fn=on_upload,
            inputs=[file_input, chatbot],
            outputs=[chatbot, file_input, doc_status],
        )

        # ── Clear chat ──────────────────────────────────────────
        def on_clear():
            return [], "*Document still loaded. Ask more questions or upload a new one.*"

        clear_btn.click(fn=on_clear, outputs=[chatbot, doc_status])

        # ── Chat handler (streaming with multi-turn) ────────────
        def respond(message: str, history: list):
            """Stream RAG answer token by token with conversation context."""
            if not (message and message.strip()):
                yield history, ""
                return

            msg = message.strip()
            history = history + [
                {"role": "user", "content": msg},
                {"role": "assistant", "content": "⏳ Retrieving context..."},
            ]
            yield history, ""

            logger.info("Processing question: '%.80s...'", msg)
            try:
                # Pass conversation history for follow-up awareness
                for chunk in stream_answer_question(msg, chat_history=history[:-2]):
                    answer = chunk["answer"]
                    if chunk["done"] and chunk["sources"]:
                        answer += "\n\n_Sources:_ " + "; ".join(
                            f"{s['source']} (chunk {s['chunk_index']})"
                            for s in chunk["sources"]
                        )
                    history[-1] = {"role": "assistant", "content": answer}
                    yield history, ""
                logger.info("Answer streamed successfully")
            except Exception as e:
                logger.exception("Error answering question")
                history[-1] = {"role": "assistant", "content": f"**Error:** {e}"}
                yield history, ""

        send_btn.click(
            fn=respond,
            inputs=[msg_input, chatbot],
            outputs=[chatbot, msg_input],
        )
        msg_input.submit(
            fn=respond,
            inputs=[msg_input, chatbot],
            outputs=[chatbot, msg_input],
        )

    return demo


def main():
    # Preload the LLM at startup so first query doesn't stall
    logger.info("Preloading LLM...")
    try:
        get_llm()
        logger.info("LLM ready")
    except Exception as e:
        logger.warning("Could not preload LLM: %s", e)

    demo = build_ui()
    demo.launch(
        server_name=config.API_HOST,
        server_port=config.API_PORT,
        share=False,
        theme=gr.themes.Soft(),
    )


if __name__ == "__main__":
    main()
