# Smart Contract Summary & QA Assistant

A small-scale web application enabling users to upload long documents (like contracts, insurance policies, or reports) and interact with them via a conversational assistant. Built with FastAPI, LangServe, LangChain, and Gradio. 

This project aligns with the NVIDIA DLI Course requirements.

## 🚀 Features
- **File Ingestion:** Supports PDF and DOCX uploads.
- **Vector Search:** Chunks and embeds documents using `SentenceTransformers` into a ChromaDB instance for fast and relevant semantic retrieval.
- **Conversational QA:** Multi-turn Q&A using local models (Llama via GGUF) or cloud alternatives (Groq, OpenAI) with rigorous prompt guardrails and source citations.
- **Document Summarization:** Dedicated pipeline to generate concise overviews of entire uploaded contracts.
- **REST APIs:** Full integration with LangServe, allowing programmatic access alongside the UI.

## 🛠 Prerequisites & Installation

### 1. Requirements
Ensure you have the following installed on your machine:
- Python 3.10+
- (Optional but recommended) CUDA toolkit for GPU acceleration if using local embedding/generation models.

### 2. Install Dependencies
Clone the repository and install the required dependencies:
```bash
pip install -r requirements.txt
```

### 3. Environment Configuration
Copy `.env.example` to `.env` and fill in your configurations:
```bash
cp .env.example .env
```
_Key settings to configure:_
- **Local Models:** Set `LOCAL_MODEL_PATH` to the absolute path of your downloaded `.gguf` model (e.g., Llama 3 8B Instruct).
- **Cloud Models (Fallback):** Set `GROQ_API_KEY` or `OPENAI_API_KEY`. The system defaults to Groq or OpenAI if no local model path is provided.
- **Server:** `API_HOST` and `API_PORT` (default is 127.0.0.1:8000).

---

## 🏃 Running the Application

### Start the Server (API + Gradio UI)
The project runs a single FastAPI server that serves both the API routes via LangServe and the Gradio UI frontend.

Run the main application using:
```bash
python api.py
```
*Note: Running `python app.py` is also supported, but `api.py` mounts both the backend endpoints and the UI correctly.*

### Accessing the Interfaces
- **Web UI:** Navigate to `http://127.0.0.1:8000/ui` to upload documents, ask questions, and summarize content.
- **API Docs:** Navigate to `http://127.0.0.1:8000/docs` to see the FastAPI interactive documentation.

---

## 📊 Evaluation

An automated evaluation pipeline is included to test the answer relevance and groundedness using an LLM-as-a-judge approach.

To run the evaluation:
1. Ensure the `data/uploads` folder contains the necessary test document (or modify the path in `scripts/evaluate.py`).
2. Run the script:
   ```bash
   python -m scripts.evaluate
   ```
The output will be saved as `Evaluation_Report.md` in the `docs/` folder.

---

## 🗂 Project Structure
- `app.py`: Gradio UI frontend definitions.
- `api.py`: FastAPI server and LangServe routes definition.
- `config.py`: Centralized configuration management and model settings.
- `pipelines/`: Core logic modules.
  - `ingestion.py`: Parsing, chunking, and embedding.
  - `retrieval.py`: Similarity search over Chroma DB.
  - `llm_pipeline.py`: Chat history management, prompt templating, LLM calling, and guardrails.
  - `summarization.py`: Aggregation and map-reduce summarization logic.
  - `vectorstore.py`: Persistent database setup.
- `docs/`: Specs, reports, and initialization prompts.
