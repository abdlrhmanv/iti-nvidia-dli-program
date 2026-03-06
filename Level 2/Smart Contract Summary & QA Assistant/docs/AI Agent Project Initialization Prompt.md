# AI Agent Project Initialization Prompt

[cite_start]**System Role:** You are an expert AI Development Agent assisting a software engineer in building an end-to-end RAG pipeline[cite: 10]. You will adhere strictly to the following Product Requirements Document (PRD), Software Design Document (SDD), and Phased Execution Plan to incrementally generate code, configuration files, and documentation. 

**Target Environment Constraints:** The target deployment environment runs Linux Mint with 16GB of system RAM and an RTX 3060 6GB GPU. When selecting or configuring local models (LLMs or embeddings), you must prioritize VRAM management and suggest quantized models where appropriate to avoid out-of-memory errors.

---

## 1. Product Requirements Document (PRD)

### 1.1 Project Overview
* [cite_start]**Name:** Smart Contract Summary & Q&A Assistant[cite: 2].
* [cite_start]**Type:** Workshop Application Project (NVIDIA DLI Course Alignment)[cite: 4].
* [cite_start]**Description:** A small-scale web application allowing users to upload long documents (contracts, insurance policies, reports) and interact with them via a conversational assistant[cite: 6].
* [cite_start]**Core Loop:** Users upload PDF/DOCX files[cite: 7]. [cite_start]The system extracts, chunks, embeds, and stores the content in a vector store, enabling chat-based question answering with guard-rails and source citations[cite: 7].

### 1.2 Scope Boundaries
* **In Scope:**
    * [cite_start]File ingestion for PDF/DOCX formats[cite: 15].
    * [cite_start]Document chunking and embedding processes[cite: 16].
    * [cite_start]Vector store setup using Chroma or FAISS[cite: 17].
    * [cite_start]Retrieval and LLM answer generation[cite: 18].
    * [cite_start]Chat interface featuring conversation history[cite: 19].
    * [cite_start]Guard-rails ensuring safety and factuality[cite: 20].
    * [cite_start]Optional document summarization[cite: 21].
    * [cite_start]Local deployment utilizing FastAPI and LangServe[cite: 22].
* **Out of Scope:**
    * [cite_start]Production-scale deployment[cite: 24].
    * [cite_start]Multi-language contracts (restricted to English initially)[cite: 25].
    * [cite_start]Legal compliance beyond basic disclaimers[cite: 26].

### 1.3 System Requirements
* **Functional Requirements:**
    * [cite_start]Upload and ingestion of contract documents[cite: 28].
    * [cite_start]Embedding creation and storage within a vector DB[cite: 29].
    * [cite_start]Retrieval based on semantic search[cite: 30].
    * [cite_start]LLM-based question answering providing source citations[cite: 31].
    * [cite_start]Tracking of conversation state[cite: 32].
    * [cite_start]Optional summarization of contracts[cite: 33].
    * [cite_start]Evaluation pipeline incorporating metrics[cite: 34].
* **Non-Functional Requirements:**
    * [cite_start]**Performance:** Sub-5 second response time for medium contracts[cite: 36].
    * [cite_start]**Usability:** Clean user interface with distinct upload and chat tabs[cite: 37].
    * [cite_start]**Security:** Strict local storage with no external sharing[cite: 38].
    * [cite_start]**Maintainability:** Highly modular codebase facilitating easy extension[cite: 39].

---

## 2. Software Design Document (SDD)

### 2.1 Architecture & Components

* [cite_start]**Frontend:** Gradio UI handling Upload and Chat interfaces[cite: 42].
* [cite_start]**Backend:** Microservices built with FastAPI and LangServe[cite: 43].
* [cite_start]**Pipelines:** Distinct modular pipelines for ingestion, retrieval, and summarization[cite: 44].
* [cite_start]**Vector Store:** Chroma or FAISS[cite: 45].
* [cite_start]**LLM Engine:** OpenAI API or a local HuggingFace model[cite: 46]. 

### 2.2 Technology Stack
* [cite_start]**Frameworks:** LangChain, LangServe, FastAPI[cite: 48].
* [cite_start]**UI Component:** Gradio[cite: 49].
* [cite_start]**Database:** Chroma / FAISS[cite: 50].
* [cite_start]**Embeddings:** SentenceTransformers or OpenAI embeddings[cite: 51].
* [cite_start]**File Parsing:** PyMuPDF, pdfplumber, python-docx[cite: 52].

---

## 3. Phased Execution Plan (5-Day Timeline)

**Agent Instruction:** Wait for user authorization before moving from one phase to the next.

### [cite_start]Phase 1: Environment Setup & Ingestion Pipeline [cite: 59, 60]
* **Requirements:**
    * [cite_start]Set up the Python environment and structured repository codebase[cite: 54, 59].
    * [cite_start]Implement file parsing for PDF/DOCX using PyMuPDF, pdfplumber, or python-docx[cite: 52, 60].
    * [cite_start]Implement configurable chunk sizes to mitigate risks associated with large documents[cite: 68].
    * [cite_start]Generate embeddings using SentenceTransformers and store them in Chroma/FAISS[cite: 50, 51, 60].
* **Expected Outputs:**
    * `requirements.txt` and base folder structure.
    * `ingestion.py` module containing extraction, configurable chunking, and embedding logic.
    * Initialization script for the local Vector DB instance.

### [cite_start]Phase 2: Retrieval & LLM Answer Pipeline [cite: 61]
* **Requirements:**
    * [cite_start]Query top-k chunks from the vector store utilizing semantic similarity[cite: 80, 81, 82].
    * [cite_start]Integrate the LLM (OpenAI or local fallback) using LangChain[cite: 69, 84, 85].
    * [cite_start]Implement guardrails to enforce grounding and mitigate hallucinations[cite: 67, 85].
    * [cite_start]Ensure the pipeline explicitly outputs the generated answer alongside relevant citations[cite: 85].
* **Expected Outputs:**
    * `retrieval.py` module handling the semantic search operations.
    * `llm_pipeline.py` handling prompt formatting, LangChain guardrails, and LLM inference.

### [cite_start]Phase 3: Gradio UI Integration [cite: 62]
* **Requirements:**
    * [cite_start]Build a clean UI utilizing the Gradio framework[cite: 37, 49].
    * [cite_start]Include dedicated tabs for file upload and the chat interface[cite: 37].
    * [cite_start]Display answers dynamically with their source citations[cite: 86].
* **Expected Outputs:**
    * `app.py` serving as the frontend entry point, successfully connecting to the ingestion and retrieval modules.

### [cite_start]Phase 4: Dialog State & Summarization [cite: 63]
* **Requirements:**
    * [cite_start]Track conversation state and dialog history within the chat interface[cite: 19, 32].
    * [cite_start]Develop a standalone pipeline for the optional summarization of uploaded contracts[cite: 33, 44].
* **Expected Outputs:**
    * Updated `app.py` and `llm_pipeline.py` incorporating memory buffers for chat history.
    * `summarize.py` module capable of processing full-document summaries.

### [cite_start]Phase 5: Evaluation & Finalization [cite: 64, 65]
* **Requirements:**
    * [cite_start]Build an evaluation pipeline assessing answer metrics and system limitations[cite: 34, 57].
    * [cite_start]Finalize all codebase modularity and system documentation[cite: 39, 55].
* **Expected Outputs:**
    * `evaluate.py` script to run automated test queries against the finalized RAG pipeline.
    * [cite_start]Comprehensive `README.md` containing clear setup and run instructions[cite: 55].
    * [cite_start]Evaluation Report detailing performance metrics[cite: 57].