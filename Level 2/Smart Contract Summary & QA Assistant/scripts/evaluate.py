"""
Evaluation pipeline.
Runs a set of test questions against the RAG pipeline and uses the LLM 
as a judge to score the results on Groundedness and Context Relevance.
Outputs the results to docs/Evaluation_Report.md.
"""

import logging
import json
import time
import sys
from pathlib import Path

# Add the project root to the python path so imports work when run as a script
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import JsonOutputParser

from pipelines.llm_pipeline import answer_question, get_llm
from pipelines.vectorstore import get_vectorstore
from pipelines.ingestion import ingest_document
import config

logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")
logger = logging.getLogger(__name__)

# A simple set of test questions covering extraction, reasoning, and out-of-domain
TEST_QA_PAIRS = [
    {
        "question": "Who are the parties entering into this contract?",
        "expected_topics": ["client", "freelancer", "names"],
    },
    {
        "question": "What is the agreed upon hourly rate or remuneration?",
        "expected_topics": ["payment", "euros", "rate", "remuneration"],
    },
    {
        "question": "Can the freelancer hire third parties to complete the work?",
        "expected_topics": ["third parties", "subcontractors", "consent"],
    },
    {
        "question": "What is the recipe for a chocolate cake?",
        "expected_topics": ["out of context", "not found"],
    }
]

JUDGE_PROMPT = """\
You are an impartial judge evaluating a RAG (Retrieval-Augmented Generation) system.
You will be given a question, the retrieved context, and the system's answer.

Please evaluate two metrics on a scale of 0 to 5:
1. Context Relevance (0-5): Does the retrieved context contain the information needed to answer the question? (If the question is out of domain, and context is empty/irrelevant, score 5 if it correctly avoided matching wrong info).
2. Answer Groundedness (0-5): Is the answer entirely supported by the context without hallucinating?

Return ONLY a valid JSON object with the exact keys "context_relevance" (int), "answer_groundedness" (int), and "reasoning" (string).

Question: {question}
Retrieved Context: {context}
System Answer: {answer}
"""

import shutil

def evaluate():
    """Runs the evaluation loop and generates a markdown report."""
    logger.info("Starting evaluation pipeline...")
    
    # Needs a document to test against (use config paths so script works from any cwd)
    sample_doc = config.UPLOAD_DIR / "vertrag-ueber-freie-mitarbeiter-englisch-data-data.pdf"
    if not sample_doc.exists():
        logger.error("Sample document not found. Upload it first via the UI or place it in %s", config.UPLOAD_DIR)
        return
        
    logger.info("Ensuring test document is ingested...")
    try:
        ingest_document(sample_doc)
    except shutil.SameFileError:
        logger.info("Document already in uploads folder, continuing.")
    
    llm = get_llm()
    judge_prompt = ChatPromptTemplate.from_messages([("human", JUDGE_PROMPT)])
    judge_chain = judge_prompt | llm | JsonOutputParser()
    
    results = []
    total_relevance = 0
    total_groundedness = 0
    
    logger.info("Running test queries...")
    for idx, test in enumerate(TEST_QA_PAIRS, 1):
        q = test["question"]
        logger.info(f"Testing Q{idx}: {q}")
        
        t0 = time.time()
        # Answer using our RAG pipeline
        rag_result = answer_question(q)
        latency = time.time() - t0
        
        answer = rag_result["answer"]
        
        # Reconstruct context text for the judge
        store = get_vectorstore()
        retrieved_docs = store.similarity_search(q, k=config.RETRIEVAL_TOP_K)
        context_str = "\n\n".join(d.page_content for d in retrieved_docs)
        
        # Ask LLM-as-a-judge to score
        try:
            score = judge_chain.invoke({
                "question": q,
                "context": context_str,
                "answer": answer
            })
            relevance = int(score.get("context_relevance", 0))
            groundedness = int(score.get("answer_groundedness", 0))
            reasoning = score.get("reasoning", "No reasoning provided.")
        except Exception as e:
            logger.error(f"Judge failed to parse JSON for Q{idx}: {e}")
            relevance = 0
            groundedness = 0
            reasoning = "Judge parsing failed."
            
        total_relevance += relevance
        total_groundedness += groundedness
            
        results.append({
            "question": q,
            "answer": answer,
            "latency_sec": round(latency, 2),
            "relevance": relevance,
            "groundedness": groundedness,
            "reasoning": reasoning,
            "sources_count": len(rag_result["sources"])
        })
        
    num_tests = len(TEST_QA_PAIRS)
    avg_rel = total_relevance / num_tests
    avg_grnd = total_groundedness / num_tests
    
    # Generate Markdown Report
    report = f"""# Evaluation Report

**Date:** {time.strftime('%Y-%m-%d')}
**Model:** `{config.LOCAL_MODEL_PATH.split('/')[-1] if config.LOCAL_MODEL_PATH else config.OPENAI_MODEL}`
**Document:** `{sample_doc.name}`

## Overall Metrics
- **Average Context Relevance:** {avg_rel:.1f}/5.0
- **Average Answer Groundedness:** {avg_grnd:.1f}/5.0
- **Total Tests Run:** {num_tests}

## Detailed Results

"""
    for i, res in enumerate(results, 1):
        report += f"### Q{i}: {res['question']}\n"
        report += f"**Answer:** {res['answer']}\n\n"
        report += f"- **Latency:** {res['latency_sec']}s\n"
        report += f"- **Sources used:** {res['sources_count']}\n"
        report += f"- **Context Relevance (0-5):** {res['relevance']}\n"
        report += f"- **Answer Groundedness (0-5):** {res['groundedness']}\n"
        report += f"- **Judge Reasoning:** *{res['reasoning']}*\n\n"
        report += "---\n\n"
        
    report += """## Limitations Identified
- Context relevance depends entirely on the embedding model (MiniLM). Complex reasoning might require better embeddings.
- LLM-as-a-judge using a 7B model can sometimes struggle to output valid JSON consistently, affecting score parsing.
- Latency is bound by local GPU performance layer offloading.
"""

    report_path = config.DOCS_DIR / "Evaluation_Report.md"
    report_path.write_text(report, encoding="utf-8")
    logger.info(f"Evaluation complete. Report saved to {report_path}")

if __name__ == "__main__":
    evaluate()
