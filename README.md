# Cost-Efficient RAG & LLM-as-Judge Evaluation

A lightweight Retrieval-Augmented Generation (RAG) system designed to answer questions using indexed documents, reduce unsupported answers, evaluate retrieval quality, and assess LLM response preferences using a pairwise LLM-as-Judge evaluator.

## 1. Project Overview

This project implements two components:

- **Problem 1 — Cost-Efficient RAG:** A document ingestion and retrieval pipeline using Gemini embeddings, FAISS vector search, and Gemini for answer generation.
- **Problem 2 — LLM-as-Judge:** A pairwise evaluation framework that compares two candidate answers, assigns scores, selects a winner, and checks whether the preference changes when answer order is reversed.

The implementation prioritizes local vector storage, reusable document indexing, configurable retrieval, and measurable evaluation.

## 2. Tech Stack

| Component | Technology |
|---|---|
| Language | Python 3.11 |
| API framework | FastAPI |
| Vector search | FAISS (`faiss-cpu`) |
| Embeddings | Google Gemini Embedding API |
| Answer generation | Google Gemini API |
| Document processing | PyPDF, BeautifulSoup, Markdown |
| Configuration | python-dotenv |
| Evaluation output | JSON |
| API testing | FastAPI Swagger UI |

## 3. Architecture

### RAG pipeline

1. **Ingestion:** Read supported PDF, Markdown, and HTML documents.
2. **Chunking:** Split document content into configurable character-based chunks with overlap.
3. **Embedding:** Generate vector embeddings using the Gemini embedding model.
4. **Indexing:** Normalize embeddings and store them in a FAISS inner-product index. Store chunk text and metadata separately in JSON.
5. **Retrieval:** Embed the user's query and retrieve the most similar indexed chunks.
6. **Grounded generation:** Send relevant context to Gemini and instruct it to answer using the supplied evidence.
7. **Abstention:** Return an insufficient-information response when no retrieved chunk meets the configured similarity threshold.
8. **Evaluation:** Measure retrieval metrics and record latency and per-question results.

### LLM-as-Judge pipeline

1. Load test cases containing a question, two candidate answers, and a reference preference.
2. Ask Gemini to score both answers on correctness, relevance, instruction following, and clarity.
3. Record the scores, winner, and explanation.
4. Swap the answer order and repeat the evaluation.
5. Map the reversed result back to the original answer labels.
6. Compare the two verdicts to detect position-preference changes.
7. Save detailed results and a summary to JSON.

## 4. Project Structure

```text
cost_efficient_rag_v2/
├── app/
│   ├── __init__.py
│   ├── config.py
│   ├── embeddings.py
│   ├── evaluate.py
│   ├── evaluate_judge.py
│   ├── ingest.py
│   ├── main.py
│   ├── rag.py
│   └── store.py
├── data/
│   ├── evaluation_questions.json
│   ├── evaluation_results.json
│   ├── judge_cases.json
│   ├── judge_results.json
│   ├── faiss.index
│   └── chunks.json
├── documents/
│   └── plant_care.md
├── .env.example
├── .gitignore
├── README.md
└── requirements.txt
```

Generated index and metadata filenames should match the paths configured in the implementation. Keep local secrets and the virtual environment outside version control.

## 5. Setup and Installation

### Prerequisites

- Python 3.11
- A Google Gemini API key with access to the configured models
- Git, if cloning the repository

### Create and activate a virtual environment

On Windows PowerShell:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

Install dependencies:

```powershell
python -m pip install -r requirements.txt
```

### Configure environment variables

Create a `.env` file in the project root. Use the following structure and fill in your own valid API key:

```dotenv
GEMINI_API_KEY=your_api_key_here
GEMINI_MODEL=gemini-3.8-flash
EMBEDDING_MODEL=gemini-embedding-2
CHUNK_SIZE=800
CHUNK_OVERLAP=120
TOP_K=5
```

The model identifiers must be available to your Gemini API project. Never commit `.env` or publish an API key.

## 6. Running the Application

### Step 1 — Ingest documents

From the project root:

```powershell
python -m app.ingest documents
```

The ingestion script processes supported documents, creates chunks, generates embeddings, and updates the local FAISS index and metadata. Re-ingesting a document should avoid adding duplicate chunk IDs where deterministic IDs are used.

### Step 2 — Start the API

```powershell
python -m uvicorn app.main:app --reload
```

The API is available at:

- Base URL: `http://127.0.0.1:8000`
- Interactive API documentation: `http://127.0.0.1:8000/docs`
- Health endpoint: `GET /api/v1/health`
- Query endpoint: `POST /api/v1/query`

The exact endpoint paths are defined by the application's route configuration.

### Step 3 — Test retrieval and answer generation

Use the Swagger interface to submit a question that should be answered by the indexed