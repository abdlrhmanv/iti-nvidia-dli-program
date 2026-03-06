"""
LLM answer pipeline: format prompts with retrieved context, enforce
grounding guardrails, call the LLM, and return an answer with citations.

Supports two providers:
  - "local"  : llama-cpp-python with a GGUF quantized model on GPU
  - "openai" : OpenAI ChatCompletion API
"""

from __future__ import annotations

import logging
import re
from typing import Optional

from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

import config
from pipelines.retrieval import retrieve_chunks, format_context

logger = logging.getLogger(__name__)

# ── Prompt Templates ─────────────────────────────────────────────

SYSTEM_PROMPT = """\
You are a precise contract analysis assistant. Your role is to answer \
questions about uploaded documents using ONLY the provided context.

RULES YOU MUST FOLLOW:
1. Answer ONLY from the context provided below. If the context does not \
contain enough information, say "I cannot find this information in the \
uploaded documents."
2. NEVER invent, assume, or hallucinate information not present in the context.
3. Cite your sources using [Source N] tags that correspond to the context \
chunks provided.
4. Keep answers concise and factual.
5. If the question is ambiguous, ask for clarification rather than guessing.
6. For legal/contractual content, include a disclaimer that this is not \
legal advice.

CONTEXT:
{context}"""

HUMAN_TEMPLATE = "{question}"

_rag_prompt = ChatPromptTemplate.from_messages([
    ("system", SYSTEM_PROMPT),
    ("human", HUMAN_TEMPLATE),
])


# ── LLM Provider ─────────────────────────────────────────────────

_llm_instance = None


def get_llm():
    """
    Lazy-load the LLM based on config.LLM_PROVIDER.
    Singleton to avoid reloading the model on every call.
    """
    global _llm_instance
    if _llm_instance is not None:
        return _llm_instance

    provider = config.LLM_PROVIDER.lower()

    if provider == "openai":
        from langchain_openai import ChatOpenAI

        if not config.OPENAI_API_KEY:
            raise ValueError(
                "LLM_PROVIDER is 'openai' but OPENAI_API_KEY is not set. "
                "Set it in your .env file or environment."
            )
        _llm_instance = ChatOpenAI(
            api_key=config.OPENAI_API_KEY,
            temperature=config.LLM_TEMPERATURE,
            max_tokens=config.LLM_MAX_TOKENS,
        )
        logger.info("Loaded OpenAI LLM (temperature=%.2f)", config.LLM_TEMPERATURE)

    elif provider == "local":
        from langchain_community.llms import LlamaCpp

        if not config.LOCAL_MODEL_PATH:
            raise ValueError(
                "LLM_PROVIDER is 'local' but LOCAL_MODEL_PATH is not set. "
                "Download a GGUF model and set the path in config or .env."
            )
        _llm_instance = LlamaCpp(
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
    else:
        raise ValueError(f"Unknown LLM_PROVIDER: {provider!r}. Use 'local' or 'openai'.")

    return _llm_instance


# ── RAG Chain ─────────────────────────────────────────────────────

def build_rag_chain():
    """Build the LangChain RAG chain: prompt | LLM | output parser."""
    llm = get_llm()
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
            sorted({d.metadata.get("source", "unknown") for d in context_docs})
        )
        answer += f"\n\n_Note: This answer is based on content from: {sources}_"

    answer += (
        "\n\n---\n"
        "_Disclaimer: This is an AI-generated summary and does not "
        "constitute legal advice. Always consult a qualified professional "
        "for legal matters._"
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
                "No relevant documents were found for your question. "
                "Please upload a document first using the ingestion pipeline."
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
