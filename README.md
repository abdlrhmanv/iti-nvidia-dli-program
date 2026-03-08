# ITI — NVIDIA DLI Program

Coursework and project materials for the **NVIDIA Deep Learning Institute (DLI)** program completed through the **Information Technology Institute (ITI)**.

## Repository Structure

```
├── Level 1/
│   └── certificates/                          # Level 1 completion certificate
│
├── Level 2/
│   ├── certificates/                          # Level 2 completion certificates
│   ├── slides/                                # DLI RAG course slides
│   ├── notebooks/                             # Hands-on course notebooks
│   │   ├── 04_running_state.ipynb
│   │   ├── 05_documents.ipynb
│   │   ├── 06_embeddings.ipynb
│   │   ├── 07_vectorstores.ipynb
│   │   ├── 08_evaluation.ipynb
│   │   └── 09_langserve.ipynb
│   └── Smart Contract Summary & QA Assistant/ # Capstone RAG project
│       ├── docs/                              # Project documentation & specs
│       │   ├── AI Agent Project Initialization Prompt.md
│       │   ├── Phase 1.md
│       │   ├── Phase 2.md
│       │   ├── Phase 3.md
│       │   ├── Phase 4 - Additional Features.md
│       │   └── Smart_Contract_Assistant_Spec.pdf
│       ├── pipelines/                         # Core RAG pipeline modules
│       │   ├── ingestion.py                   # Document ingestion pipeline
│       │   ├── retrieval.py                   # Semantic search & retrieval
│       │   ├── llm_pipeline.py                # LLM answer generation with guardrails
│       │   ├── summarization.py               # Full-document summarization
│       │   └── vectorstore.py                 # ChromaDB vector store helpers
│       ├── scripts/                           # Utility scripts
│       │   ├── init_vectordb.py                # Vector DB initialization & reset
│       │   └── evaluate.py                    # RAG evaluation (LLM-as-judge)
│       ├── app.py                             # Gradio UI (standalone)
│       ├── api.py                             # FastAPI + LangServe + Gradio at /ui
│       ├── config.py                          # Centralized configuration
│       ├── requirements.txt                   # Python dependencies
│       └── .env.example                       # Environment variable template
│
├── .gitignore
└── README.md
```

## Level 1 — Fundamentals

Introductory DLI course covering core concepts of deep learning and AI.

## Level 2 — Building RAG Applications

Advanced course focused on **Retrieval-Augmented Generation (RAG)** pipelines, covering:

- Document loading and text extraction
- Text chunking strategies
- Embedding generation with SentenceTransformers
- Vector store operations with ChromaDB
- Evaluation metrics for RAG systems
- API serving with LangServe

### Capstone Project — Smart Contract Summary & Q&A Assistant

A local RAG application that lets users upload contracts (PDF/DOCX) and ask questions about them via a conversational interface. 

For full details, setup instructions, and architecture overview, please see the **[Project Documentation](./Level%202/Smart%20Contract%20Summary%20&%20QA%20Assistant/README.md)**.

## License

This repository contains personal coursework and is not licensed for redistribution.
