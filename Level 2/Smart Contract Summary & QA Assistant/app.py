"""
Gradio frontend for the Smart Contract Summary & QA Assistant.

Phase 3: UI with dedicated tabs for file upload and chat, displaying
answers with source citations. Connects to ingestion and LLM pipelines.
"""

from __future__ import annotations

import logging
from pathlib import Path

import gradio as gr

import config
from pipelines.ingestion import ingest_document
from pipelines.llm_pipeline import answer_question

logging.basicConfig(
    level=logging.INFO,
    format="%(levelname)s | %(name)s | %(message)s",
)
logger = logging.getLogger(__name__)

# Ensure upload/vectorstore dirs exist for Gradio file handling
config.ensure_dirs()


def handle_upload(file) -> str:
    """
    Process an uploaded PDF or DOCX: ingest into the vector store and
    return a user-friendly status message.
    """
    if file is None:
        return "Please select a PDF or DOCX file to upload."
    # Gradio File can return a single path (str), a path object, or a list
    if isinstance(file, (list, tuple)) and file:
        file = file[0]
    path = Path(file.name) if hasattr(file, "name") else Path(file)
    if not path.exists():
        return f"File not found: {path}"

    ext = path.suffix.lower()
    if ext not in config.SUPPORTED_EXTENSIONS:
        return (
            f"Unsupported file type: **{ext}**. "
            f"Supported: **{', '.join(config.SUPPORTED_EXTENSIONS)}**"
        )

    try:
        result = ingest_document(path)
        return (
            f"**Upload successful**\n\n"
            f"- **File:** {result['filename']}\n"
            f"- **Characters extracted:** {result['characters']:,}\n"
            f"- **Chunks created:** {result['chunks']}\n"
            f"- **Chunk size / overlap:** {result['chunk_size']} / {result['chunk_overlap']}\n\n"
            f"You can now ask questions about this document in the **Chat** tab."
        )
    except ValueError as e:
        return f"**Error:** {e}"
    except Exception as e:
        logger.exception("Ingestion failed")
        return f"**Error:** {e}"


def handle_question(question: str) -> tuple[str, str]:
    """
    Run the RAG pipeline and return (answer_markdown, sources_markdown).
    """
    if not (question and question.strip()):
        return (
            "Please enter a question about your uploaded documents.",
            "",
        )

    try:
        result = answer_question(question.strip())
        answer = result["answer"]
        sources = result["sources"]

        # Format sources for display
        if sources:
            lines = [
                f"- **{s['source']}** (chunk {s['chunk_index']}, "
                f"relevance: {s['relevance_score']:.4f})"
                for s in sources
            ]
            sources_md = "**Sources:**\n\n" + "\n".join(lines)
        else:
            sources_md = ""

        return answer, sources_md
    except ValueError as e:
        return f"**Error:** {e}", ""
    except Exception as e:
        logger.exception("Answer pipeline failed")
        return f"**Error:** {e}", ""


def build_ui():
    """Build the Gradio interface with Upload and Chat tabs."""
    with gr.Blocks(
        title="Smart Contract Summary & QA Assistant",
        theme=gr.themes.Soft(),
        css="""
        .citation { font-size: 0.9em; color: #555; margin-top: 0.5em; }
        """,
    ) as demo:
        gr.Markdown(
            "# Smart Contract Summary & QA Assistant\n"
            "Upload contract documents (PDF/DOCX), then ask questions in the Chat tab. "
            "Answers are grounded in your documents with source citations."
        )

        with gr.Tabs():
            # ── Tab 1: Upload ─────────────────────────────────────
            with gr.TabItem("Upload"):
                gr.Markdown(
                    "Upload a **PDF** or **DOCX** contract. The document will be "
                    "chunked, embedded, and stored for question-answering."
                )
                upload_input = gr.File(
                    label="Document",
                    file_types=[".pdf", ".docx"],
                    type="filepath",
                )
                upload_btn = gr.Button("Ingest document", variant="primary")
                upload_output = gr.Markdown(
                    label="Status",
                    value="Select a file and click **Ingest document**.",
                )
                upload_btn.click(
                    fn=handle_upload,
                    inputs=[upload_input],
                    outputs=[upload_output],
                )

            # ── Tab 2: Chat ───────────────────────────────────────
            with gr.TabItem("Chat"):
                gr.Markdown(
                    "Ask a question about your uploaded documents. "
                    "Answers include **source citations** and a relevance-ranked source list."
                )
                question_input = gr.Textbox(
                    label="Question",
                    placeholder="e.g. What is the termination clause?",
                    lines=2,
                )
                chat_btn = gr.Button("Get answer", variant="primary")
                answer_output = gr.Markdown(label="Answer")
                sources_output = gr.Markdown(label="Sources", visible=True)
                chat_btn.click(
                    fn=handle_question,
                    inputs=[question_input],
                    outputs=[answer_output, sources_output],
                )

    return demo


def main():
    demo = build_ui()
    demo.launch(
        server_name=config.API_HOST,
        server_port=config.API_PORT,
        share=False,
    )


if __name__ == "__main__":
    main()
