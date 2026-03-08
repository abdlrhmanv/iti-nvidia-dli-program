"""
Summarization pipeline: extracts all chunks from the vector store
and generates a comprehensive summary of the uploaded document.
"""

import logging
from typing import Any, Optional

from langchain_classic.chains.summarize import load_summarize_chain
from langchain_core.callbacks import BaseCallbackHandler
from langchain_core.documents import Document
import config
from pipelines.vectorstore import get_docstore, get_retriever
from pipelines.llm_pipeline import get_llm

logger = logging.getLogger(__name__)


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
    
    # If it fits in context, use 'stuff' (single LLM call). Otherwise map_reduce.
    max_stuff_tokens = config.LLM_N_CTX - config.LLM_MAX_TOKENS - 500
    
    if estimated_tokens <= max_stuff_tokens:
        chain_type = "stuff"
        docs_to_summarize = docs
        logger.info("Summarizing %d chunks (~%d tokens) in one step (stuff)", len(docs), estimated_tokens)
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

    try:
        chain = load_summarize_chain(llm, chain_type=chain_type)
        callbacks = [_SummarizeProgressHandler(len(docs_to_summarize))] if chain_type == "map_reduce" else None
        summary = chain.invoke(docs_to_summarize, config={"callbacks": callbacks} if callbacks else {})
        
        # Depending on chain_type, result is either in 'output_text' or direct string
        result = summary.get("output_text", str(summary))
        logger.info("Summarization complete")
        return result
    except Exception as e:
        logger.exception("Summarization failed")
        return f"Failed to summarize document: {e}"

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")
    print("Generating summary...")
    print(summarize_document())
