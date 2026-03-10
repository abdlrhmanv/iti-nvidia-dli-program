"""
LLM answer pipeline: format prompts with retrieved context, enforce
grounding guardrails, call the LLM, and return an answer with citations.

Uses a local GGUF model (llama-cpp-python) when LOCAL_MODEL_PATH is set,
otherwise falls back to OpenAI if OPENAI_API_KEY is set.
"""

from __future__ import annotations

import logging
import re
from pathlib import Path
from typing import Optional

from langchain_community.chat_models.llamacpp import ChatLlamaCpp
from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

import config
from pipelines.retrieval import retrieve_chunks, format_context

logger = logging.getLogger(__name__)

# ── Prompt Templates (tuned for natural, human-like answers) ──────

RAG_PROMPT_TEMPLATE = """\
You are an expert legal and contractual assistant responding to user inquiries based strictly on the provided document excerpts.

Guidelines:
- **Cite Sources explicitly**: Incorporate citations inline exactly where the information is stated using the format [Source N] (e.g., "The termination period is 30 days [Source 2].").
- **Clarity and Structure**: Use short, professional sentences. Break your response into readable paragraphs. If listing requirements, clauses, or multiple points, ALWAYS use bullet points for readability. DO NOT output a single wall of text.
- **Strict Grounding**: Base your answer ONLY on the excerpts below. If the answer cannot be found in the provided text, state clearly: "I cannot find the answer to this question in the provided document." Do not hallucinate or guess.
- **Completeness**: Provide a comprehensive yet concise answer covering all necessary details from the text.
- **Proportional Depth**: Your answer should be proportional to the amount of relevant information in the context. If the context contains multiple relevant clauses, sections, or provisions, address each one thoroughly rather than giving a brief overview. A question with rich context deserves a detailed, multi-paragraph answer.
- **Follow-up Questions**: At the very end of your response, provide 3 suggested follow-up questions the user could ask to learn more about this topic, formatted as a bulleted list titled '💡 **Suggested Follow-up Questions:**'

DOCUMENT EXCERPTS:
{context}

USER QUESTION: {question}

ANSWER:"""

RAG_PROMPT_WITH_HISTORY = """\
You are an expert legal and contractual assistant responding to user inquiries. Below are the relevant document excerpts, followed by the recent conversation history and the user's latest question.

Guidelines:
- **Cite Sources explicitly**: Incorporate citations inline exactly where the information is stated using the format [Source N] (e.g., "The termination period is 30 days [Source 2].").
- **Clarity and Structure**: Use short, professional sentences. Break your response into readable paragraphs. If listing requirements, clauses, or multiple points, ALWAYS use bullet points. DO NOT output a single wall of text.
- **Strict Grounding**: Base your answer ONLY on the excerpts provided. Use the conversation history solely to understand context (e.g., resolving pronouns like "it" or referencing earlier parts of the discussion). Do not hallucinate outside the given text.
- **Completeness**: Provide a comprehensive yet concise answer covering all necessary details from the text.
- **Proportional Depth**: Your answer should be proportional to the amount of relevant information in the context. If the context contains multiple relevant clauses, sections, or provisions, address each one thoroughly rather than giving a brief overview. A question with rich context deserves a detailed, multi-paragraph answer.
- **Follow-up Questions**: At the very end of your response, provide 3 suggested follow-up questions the user could ask to learn more about this topic, formatted as a bulleted list titled '💡 **Suggested Follow-up Questions:**'

DOCUMENT EXCERPTS:
{context}

CONVERSATION HISTORY:
{history}

USER QUESTION: {question}

ANSWER:"""

_rag_prompt = ChatPromptTemplate.from_messages([
    ("human", RAG_PROMPT_TEMPLATE),
])


def _build_prompt(context: str, question: str, history: str = ""):
    """Select the right prompt template based on whether history exists."""
    if history.strip():
        template = ChatPromptTemplate.from_messages([
            ("human", RAG_PROMPT_WITH_HISTORY),
        ])
        return template.format_messages(
            context=context, question=question, history=history
        )
    else:
        template = ChatPromptTemplate.from_messages([
            ("human", RAG_PROMPT_TEMPLATE),
        ])
        return template.format_messages(context=context, question=question)


# ── LLM (Local GGUF or OpenAI fallback) ───────────────────────────

_llm_instance = None


def _use_local_llm() -> bool:
    """True if a local GGUF model path is set and the file exists."""
    if not config.LOCAL_MODEL_PATH:
        return False
    return Path(config.LOCAL_MODEL_PATH).expanduser().exists()


def get_llm():
    """
    Lazy-load the LLM (singleton).
    Order: local GGUF (if path set) → Groq (if GROQ_API_KEY set) → OpenAI (if OPENAI_API_KEY set).
    """
    global _llm_instance
    if _llm_instance is not None:
        return _llm_instance

    if _use_local_llm():
        _llm_instance = ChatLlamaCpp(
            model_path=config.LOCAL_MODEL_PATH,
            n_ctx=config.LLM_N_CTX,
            n_gpu_layers=config.LLM_N_GPU_LAYERS,
            temperature=config.LLM_TEMPERATURE,
            max_tokens=config.LLM_MAX_TOKENS,
            verbose=False,
        )
        logger.info(
            "Loaded local LLM from %s (ctx=%d, gpu_layers=%d)",
            config.LOCAL_MODEL_PATH, config.LLM_N_CTX, config.LLM_N_GPU_LAYERS,
        )
        return _llm_instance

    if config.GROQ_API_KEY:
        from langchain_groq import ChatGroq
        _llm_instance = ChatGroq(
            model=config.GROQ_MODEL,
            api_key=config.GROQ_API_KEY,
            temperature=config.LLM_TEMPERATURE,
            max_tokens=config.LLM_MAX_TOKENS,
        )
        logger.info("Using Groq model: %s", config.GROQ_MODEL)
        return _llm_instance

    if config.OPENAI_API_KEY:
        from langchain_openai import ChatOpenAI
        _llm_instance = ChatOpenAI(
            model=config.OPENAI_MODEL,
            api_key=config.OPENAI_API_KEY,
            temperature=config.LLM_TEMPERATURE,
            max_tokens=config.LLM_MAX_TOKENS,
        )
        logger.info("Using OpenAI model: %s", config.OPENAI_MODEL)
        return _llm_instance

    raise ValueError(
        "No LLM configured. Set one of in .env:\n"
        "  1. LOCAL_MODEL_PATH (GGUF file), or\n"
        "  2. GROQ_API_KEY (e.g. llama-3.1-8b-instant), or\n"
        "  3. OPENAI_API_KEY (e.g. gpt-4o-mini)."
    )


# ── RAG Chain ─────────────────────────────────────────────────────

def build_rag_chain():
    """Build the LangChain RAG chain: prompt | LLM | output parser."""
    llm = get_llm()
    # Optional: use a smaller max_tokens for Q&A to speed up replies (summarization keeps LLM_MAX_TOKENS)
    qa_max = config.LLM_MAX_TOKENS_QA or config.LLM_MAX_TOKENS
    if config.LLM_MAX_TOKENS_QA:
        llm = llm.bind(max_tokens=config.LLM_MAX_TOKENS_QA)
    return _rag_prompt | llm | StrOutputParser()


# ── Guardrails ────────────────────────────────────────────────────

def apply_output_guardrails(answer: str, context_docs: list[Document]) -> str:
    """
    Post-generation guardrails to enforce grounding.

    Checks:
    1. If the answer is empty or too short, return a fallback.
    2. If the answer mentions information without any [Source N] citation,
       append a warning.
    3. Append a legal disclaimer for contract-related content.
    """
    if not answer or len(answer.strip()) < 10:
        return (
            "I was unable to generate a meaningful answer from the provided "
            "documents. Please try rephrasing your question."
        )

    has_citations = bool(re.search(r"\[Source\s*\d+\]", answer))
    if not has_citations and context_docs:
        sources = ", ".join(
            sorted({f"`{d.metadata.get('source', 'unknown')}`" for d in context_docs})
        )
        answer += f"\n\n> ℹ️ *Note: This answer is based on content from: {sources}*"

    answer += (
        "\n\n> ⚠️ *Disclaimer: This is an AI-generated response and does not "
        "constitute legal advice. Always consult a qualified professional "
        "for legal matters.*"
    )

    return answer


# ── Public Entry Points ───────────────────────────────────────────

def answer_question(
    question: str,
    top_k: Optional[int] = None,
) -> dict:
    """
    Full RAG pipeline:
      1. Retrieve relevant chunks from the vector store.
      2. Format them into a grounded prompt.
      3. Call the LLM.
      4. Apply output guardrails.

    Returns a dict with:
      - answer: the final (guardrailed) response
      - sources: list of {source, chunk_index, relevance_score} dicts
      - raw_answer: the LLM output before guardrails
    """
    context_docs = retrieve_chunks(question, top_k=top_k)

    if not context_docs:
        return {
            "answer": (
                "👋 **Welcome! I'm the Smart Contract Summary & QA Assistant.**\n\n"
                "I specialize in analyzing and answering questions about legal and contractual documents. "
                "Here's what I can do for you:\n\n"
                "- 📄 **Summarize** uploaded documents\n"
                "- ❓ **Answer questions** grounded in the document's content\n"
                "- 🔍 **Cite sources** so you can verify every claim\n\n"
                "---\n\n"
                "⚠️ **No document is currently loaded.** "
                "Please navigate to the **Upload Document** tab and upload a **PDF** or **DOCX** file to get started."
            ),
            "sources": [],
            "raw_answer": "",
        }

    context_str = format_context(context_docs)

    chain = build_rag_chain()
    raw_answer = chain.invoke({
        "context": context_str,
        "question": question,
    })

    final_answer = apply_output_guardrails(raw_answer, context_docs)

    sources = [
        {
            "source": doc.metadata.get("source", "unknown"),
            "chunk_index": doc.metadata.get("chunk_index", -1),
            "relevance_score": doc.metadata.get("relevance_score", 0.0),
        }
        for doc in context_docs
    ]

    logger.info(
        "Answered question with %d sources: '%.80s...'",
        len(sources), question,
    )

    return {
        "answer": final_answer,
        "sources": sources,
        "raw_answer": raw_answer,
    }


def stream_answer_question(
    question: str,
    top_k: Optional[int] = None,
    chat_history: list[dict] | None = None,
):
    """
    Streaming RAG pipeline — yields partial results as tokens arrive.
    Supports multi-turn conversation via chat_history.

    Yields dicts with:
      - answer: the accumulated answer so far
      - sources: list of {source, chunk_index, relevance_score} dicts
      - done: True on the final yield (after guardrails applied)
    """
    context_docs = retrieve_chunks(question, top_k=top_k)

    sources = [
        {
            "source": doc.metadata.get("source", "unknown"),
            "chunk_index": doc.metadata.get("chunk_index", -1),
            "relevance_score": doc.metadata.get("relevance_score", 0.0),
        }
        for doc in context_docs
    ]

    if not context_docs:
        yield {
            "answer": (
                "👋 **Welcome! I'm the Smart Contract Summary & QA Assistant.**\n\n"
                "I specialize in analyzing and answering questions about legal and contractual documents. "
                "Here's what I can do for you:\n\n"
                "- 📄 **Summarize** uploaded documents\n"
                "- ❓ **Answer questions** grounded in the document's content\n"
                "- 🔍 **Cite sources** so you can verify every claim\n\n"
                "---\n\n"
                "⚠️ **No document is currently loaded.** "
                "Please navigate to the **Upload Document** tab and upload a **PDF** or **DOCX** file to get started."
            ),
            "sources": [],
            "done": True,
        }
        return

    context_str = format_context(context_docs)

    # Build conversation history string from recent exchanges
    history_str = ""
    if chat_history:
        pairs = []
        for entry in chat_history:
            role = entry.get("role", "")
            content = entry.get("content", "")
            
            # Gradio can pass files as lists/tuples in the content field
            if not isinstance(content, str):
                continue
                
            if role == "user":
                pairs.append(f"User: {content}")
            elif role == "assistant" and not content.startswith("✅ I've loaded"):
                # Skip system messages (upload confirmations)
                pairs.append(f"Assistant: {content}")
        # Keep only last 6 exchanges to avoid context overflow
        history_str = "\n".join(pairs[-6:])

    prompt = _build_prompt(context_str, question, history_str)
    llm = get_llm()

    raw_answer = ""
    for chunk in llm.stream(prompt):
        token = chunk.content if hasattr(chunk, "content") else str(chunk)
        raw_answer += token
        yield {"answer": raw_answer, "sources": sources, "done": False}

    final_answer = apply_output_guardrails(raw_answer, context_docs)
    logger.info(
        "Streamed answer with %d sources: '%.80s...'",
        len(sources), question,
    )
    yield {
        "answer": final_answer, 
        "sources": sources, 
        "done": True, 
        "context_text": [d.page_content for d in context_docs]
    }


# ── CLI helper ───────────────────────────────────────────────────

if __name__ == "__main__":
    import sys

    logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")

    if len(sys.argv) < 2:
        print("Usage: python -m pipelines.llm_pipeline <question> [top_k]")
        sys.exit(1)

    question = sys.argv[1]
    k = int(sys.argv[2]) if len(sys.argv) > 2 else None

    result = answer_question(question, top_k=k)

    print("\n" + "=" * 60)
    print("ANSWER:")
    print("=" * 60)
    print(result["answer"])
    print("\nSOURCES:")
    for s in result["sources"]:
        print(f"  - {s['source']} (chunk {s['chunk_index']}, score {s['relevance_score']})")
