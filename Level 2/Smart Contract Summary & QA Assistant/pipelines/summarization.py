"""
Summarization pipeline: extracts all chunks from the vector store
and generates a comprehensive summary of the uploaded document.

The summary length scales with the document size so that large documents
receive proportionally detailed summaries.
"""

import logging
from typing import Any, Optional

from langchain_classic.chains.summarize import load_summarize_chain
from langchain_core.callbacks import BaseCallbackHandler
from langchain_core.documents import Document
from langchain_core.prompts import PromptTemplate
import config
from pipelines.vectorstore import get_docstore, get_retriever
from pipelines.llm_pipeline import get_llm

logger = logging.getLogger(__name__)


# ── Custom Prompt Templates ──────────────────────────────────────

# Map prompt: summarize each individual chunk in detail
MAP_PROMPT_TEMPLATE = """\
You are a meticulous legal document analyst. Write a **detailed summary** of the following section of a document.

Rules:
- Cover ALL key points, clauses, definitions, obligations, rights, conditions, and exceptions mentioned.
- Preserve important details such as specific terms, timeframes, monetary amounts, party names, and conditions.
- Use structured formatting: headings, bullet points, and numbered lists where appropriate.
- Do NOT omit information. Be thorough — it is better to be too detailed than too brief.
- Write at least 150-300 words for this section.

DOCUMENT SECTION:
"{text}"

DETAILED SECTION SUMMARY:"""

# Combine/reduce prompt: merge partial summaries into a final comprehensive summary
COMBINE_PROMPT_TEMPLATE = """\
You are a meticulous legal document analyst. You are given several detailed summaries of different sections of the same document. Your task is to combine them into ONE comprehensive, well-structured final summary.

Rules:
- The final summary must be **proportional to the document's length and complexity**. Target approximately {target_words} words.
- Cover ALL major sections, clauses, provisions, and topics from the partial summaries.
- Use clear **section headings** (e.g., "## Parties & Scope", "## Payment Terms", "## Termination", "## Liability", etc.) to organize the summary.
- Under each heading, use bullet points to list key provisions, obligations, rights, and conditions.
- Preserve important specifics: names, dates, amounts, percentages, defined terms.
- Remove redundancy between sections but do NOT drop any unique information.
- End with a brief "Key Takeaways" section highlighting the most critical points.
- Write in professional, clear language.

PARTIAL SUMMARIES:
"{text}"

COMPREHENSIVE FINAL SUMMARY:"""

# Stuff prompt: for small documents that fit in a single call
STUFF_PROMPT_TEMPLATE = """\
You are a meticulous legal document analyst. Write a **comprehensive, detailed summary** of the following document.

Rules:
- The summary must be **proportional to the document's length**. Target approximately {target_words} words.
- Cover ALL major sections, clauses, provisions, definitions, obligations, rights, and conditions.
- Use clear **section headings** (e.g., "## Parties & Scope", "## Key Definitions", "## Obligations", etc.).
- Under each heading, use bullet points to list key provisions.
- Preserve important specifics: names, dates, amounts, defined terms, conditions, and exceptions.
- End with a brief "Key Takeaways" section.
- Write in professional, clear language. Be thorough — do NOT produce a short paragraph.

DOCUMENT:
"{text}"

COMPREHENSIVE SUMMARY:"""


def _estimate_target_words(total_chars: int) -> int:
    """
    Estimate a target summary word count proportional to the document size.
    
    Rough heuristic:
    - Very short docs (<5K chars):  ~150 words
    - Medium docs (5K-20K chars):   ~300-500 words
    - Long docs (20K-100K chars):   ~500-1000 words
    - Very long docs (100K+ chars): ~1000-1500 words
    """
    if total_chars < 5_000:
        return 150
    elif total_chars < 20_000:
        return 300 + int((total_chars - 5_000) / 75)
    elif total_chars < 100_000:
        return 500 + int((total_chars - 20_000) / 160)
    else:
        return min(1500, 1000 + int((total_chars - 100_000) / 200))


class _SummarizeProgressHandler(BaseCallbackHandler):
    """Logs progress during map_reduce so the user sees activity."""

    def __init__(self, total_chunks: int):
        self.total_chunks = total_chunks
        self.step = 0

    def on_llm_start(self, serialized: dict, prompts: list[str], **kwargs: Any) -> None:
        self.step += 1
        if self.step <= self.total_chunks:
            logger.info("Summarization: processing chunk %d/%d...", self.step, self.total_chunks)
        else:
            logger.info("Summarization: combining results (final step)...")

def summarize_document() -> str:
    """
    Fetches all parent chunks from the docstore and generates a summary.
    If the document is small enough, it uses a 'stuff' chain. 
    Otherwise, it uses a 'map_reduce' approach.
    
    Summary length scales with document size.
    """
    docstore = get_docstore()
    
    # docstore is a LocalFileStore yielding bytes.
    # MultiVectorRetriever stores Documents directly, but picking/unpickling 
    # might be involved if LocalFileStore is strictly bytes.
    # However, we can simply ask the retriever to yield the values.
    # We will use the retrieve_chunks logic to get all documents.
    retriever = get_retriever()
    
    # LocalFileStore yields keys via yield_keys()
    keys = list(docstore.yield_keys())
    if not keys:
        return "No document loaded. Please upload a document first."

    # Mget gets the LangChain Document objects back
    docs = docstore.mget(keys)
    docs = [d for d in docs if d is not None]

    docs.sort(key=lambda d: d.metadata.get("chunk_index", 0))
    
    if not docs:
        return "No document loaded. Please upload a document first."

    llm = get_llm()
    
    # Calculate rough token count (1 token ~= 4 chars)
    total_chars = sum(len(d.page_content) for d in docs)
    estimated_tokens = int(total_chars / 4)
    target_words = _estimate_target_words(total_chars)
    
    logger.info(
        "Document stats: %d chars, ~%d tokens, target summary ~%d words",
        total_chars, estimated_tokens, target_words,
    )
    
    # If it fits in context, use 'stuff' (single LLM call). Otherwise map_reduce.
    max_stuff_tokens = config.LLM_N_CTX - config.LLM_MAX_TOKENS - 500
    
    if estimated_tokens <= max_stuff_tokens:
        chain_type = "stuff"
        docs_to_summarize = docs
        logger.info("Summarizing %d chunks (~%d tokens) in one step (stuff)", len(docs), estimated_tokens)
        
        stuff_prompt = PromptTemplate(
            template=STUFF_PROMPT_TEMPLATE.format(target_words=target_words),
            input_variables=["text"],
        )
        chain = load_summarize_chain(llm, chain_type=chain_type, prompt=stuff_prompt)
    else:
        chain_type = "map_reduce"
        # Merge small chunks into larger "super-chunks" so we do fewer map steps (much faster).
        # Target: at most ~5–6 map steps. Max ~1500 tokens per super-chunk to stay safe.
        max_chars_per_super = min(6000, (max_stuff_tokens * 4) // 2)  # ~3000 tokens per map
        super_docs = []
        current_text: list[str] = []
        current_len = 0
        batch_start_idx = 0
        for i, d in enumerate(docs):
            content = d.page_content
            if current_len + len(content) > max_chars_per_super and current_text:
                super_docs.append(Document(
                    page_content="\n\n".join(current_text),
                    metadata={"source": docs[batch_start_idx].metadata.get("source", "document")},
                ))
                batch_start_idx = i
                current_text = []
                current_len = 0
            current_text.append(content)
            current_len += len(content)
        if current_text:
            super_docs.append(Document(
                page_content="\n\n".join(current_text),
                metadata={"source": docs[batch_start_idx].metadata.get("source", "document")},
            ))
        docs_to_summarize = super_docs
        logger.info(
            "Summarizing %d chunks merged into %d super-chunks (map_reduce) — ~%d tokens",
            len(docs), len(docs_to_summarize), estimated_tokens,
        )
        logger.info("Map-reduce: %d map steps + 1 reduce step.", len(docs_to_summarize))
        
        map_prompt = PromptTemplate(
            template=MAP_PROMPT_TEMPLATE,
            input_variables=["text"],
        )
        combine_prompt = PromptTemplate(
            template=COMBINE_PROMPT_TEMPLATE.format(target_words=target_words),
            input_variables=["text"],
        )
        chain = load_summarize_chain(
            llm,
            chain_type=chain_type,
            map_prompt=map_prompt,
            combine_prompt=combine_prompt,
        )

    try:
        callbacks = [_SummarizeProgressHandler(len(docs_to_summarize))] if chain_type == "map_reduce" else None
        summary = chain.invoke(docs_to_summarize, config={"callbacks": callbacks} if callbacks else {})
        
        # Depending on chain_type, result is either in 'output_text' or direct string
        result = summary.get("output_text", str(summary))
        logger.info("Summarization complete (~%d words produced)", len(result.split()))
        return result
    except Exception as e:
        logger.exception("Summarization failed")
        return f"Failed to summarize document: {e}"

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")
    print("Generating summary...")
    print(summarize_document())
