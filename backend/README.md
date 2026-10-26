# Backend — HR Policy Bot

This document describes the backend modules and commands for ingestion, retrieval, query-routing, LCEL RAG, cross-encoder reranking, and the FastAPI application. Project-wide architecture and submission status are in the [root README](../README.md).

## Backend layout

```text
backend/
├── README.md
├── requirements.txt
├── app/
│   ├── main.py                    # FastAPI application (/api/upload, /api/query, /health)
│   ├── ingestion/
│   │   ├── pdf_loader.py          # Loads page-level documents from data/documents
│   │   ├── chunker.py             # Splits documents into overlapping chunks
│   │   └── upload_processor.py    # In-memory processor & FAISS indexer for user-uploaded PDFs
│   ├── retrieval/
│   │   ├── embeddings.py          # Local embeddings (all-MiniLM-L6-v2)
│   │   ├── faiss_store.py         # Builds and persists FAISS store
│   │   ├── query_index.py         # Top-k query with L2 distances
│   │   └── retrieval_checks.py    # Consistency and sensitivity verification
│   └── rag/
│       ├── main.py                # LCEL RAG chain (routing, retrieval, rerank, generation)
│       ├── query_router.py        # Question classifier and sub-query generator
│       └── reranker.py            # Cross-encoder reranker (ms-marco-MiniLM-L-6-v2)
└── data/
    └── index/
        └── hr_policy_faiss/       # Pre-computed FAISS index for sample documents
```

## Module responsibilities and status

| Module | Responsibility | Verified status |
|---|---|---|
| `app/main.py` | FastAPI server exposing `/health`, `/api/upload`, and `/api/query` | Implemented with CORS and active |
| `app/ingestion/upload_processor.py` | Parses uploaded PDF bytes, chunks, and creates in-memory vector store | Implemented (zero local disk persistence) |
| `app/ingestion/pdf_loader.py` | Loads PDFs page by page using `PyPDFLoader`, preserving source and zero-based page metadata | Tested: 39 pages from 2 PDFs |
| `app/ingestion/chunker.py` | Splits page-level documents and assigns `chunk_id` metadata | Tested: 124 chunks |
| `app/retrieval/embeddings.py` | Creates local embeddings with `sentence-transformers/all-MiniLM-L6-v2` | Tested: 384 dimensions per vector |
| `app/retrieval/faiss_store.py` | Builds and persists the FAISS vector store | Tested; index saved to `backend/data/index/hr_policy_faiss/` |
| `app/retrieval/query_index.py` | Loads the saved index and returns top-k chunks with source, page, chunk ID, and L2 distance | Tested |
| `app/retrieval/retrieval_checks.py` | Checks repeat-query consistency and result variation between questions | Both tests passed |
| `app/rag/query_router.py` | Classifies question type; asks LLM for structured JSON routing with deterministic fallback | Multi-part decomposition integrated; tested with fallback |
| `app/rag/reranker.py` | Cross-encoder scores FAISS candidate chunks and records before/after ranks | Standalone and integrated reranking tested |
| `app/rag/main.py` | LCEL chain for routing, FAISS retrieval, reranking, prompt/LLM answer generation, and source metadata | Integrated run produced correct multi-part answer and ranking evidence |

## Environment and configuration

Run commands from the repository root so imports such as `backend.app...` resolve consistently.

```bash
source .venv/bin/activate
python -m pip install -r backend/requirements.txt
```

The current RAG chain uses `langchain-openrouter` and `python-dotenv`, alongside the project's LangChain, `sentence-transformers`, FAISS, and PDF packages.

Configure a project-root `.env` file:

```dotenv
OPENROUTER_API_KEY=your_openrouter_api_key
OPENROUTER_MODEL=nvidia/nemotron-3.5-lightning:free
```

`app/rag/main.py` loads `.env` from the repository root. Model selection is configurable. Do not commit `.env` or real API credentials.

## Documents and index

### Do I need to save PDFs locally?

**No, you do not need to save PDFs locally.**

- **Web App / API Uploads (`POST /api/upload`)**: When uploading a PDF through the web interface or API, the document is processed completely **in memory**. A temporary directory is used only for text extraction and is immediately deleted. Chunks and FAISS vectors reside in memory for the active session (`UPLOADS` cache). No files are saved to disk.
- **Default Sample Documents (`data/documents/`)**: For running offline evaluation or CLI scripts, the repository already comes pre-bundled with sample HR policy PDFs under the root `data/documents/` folder and a pre-built FAISS index under `backend/data/index/hr_policy_faiss/`. You do not need to download or manually save anything.

The pre-bundled source PDFs are located in:

```text
data/documents/
```

The saved FAISS index is stored in:

```text
backend/data/index/hr_policy_faiss/
```

The saved store normally includes `index.faiss` and `index.pkl`. The vector store is loaded and reused by `query_index.py` and `rag/main.py`; the documents do not need to be re-embedded on each query. A new query still needs an embedding to search the index.

The current source set comprises `Veer_IT_Solutions_HR_Manual.pdf` (31 pages) and `Veer_IT_Solutions_India_Leave_Policy.pdf` (8 pages), totaling 39 page-level documents and 124 chunks.

## Commands

Run these from the repository root with `.venv` activated.

```bash
# Start the FastAPI server (recommended for web app)
uvicorn backend.app.main:app --reload --port 8000

# Load page-level policy documents (CLI test)
python -m backend.app.ingestion.pdf_loader

# Chunk documents
python -m backend.app.ingestion.chunker

# Generate local embeddings
python -m backend.app.retrieval.embeddings

# Build and persist FAISS
python -m backend.app.retrieval.faiss_store

# Query the persisted index
python -m backend.app.retrieval.query_index

# Check repeatability and question sensitivity
python -m backend.app.retrieval.retrieval_checks

# Run rule-based router examples
python -m backend.app.rag.query_router

# Run standalone reranking comparison
python -m backend.app.rag.reranker

# Run the integrated RAG chain with routing and reranking
python -m backend.app.rag.main
```

## Retrieval behavior

`query_index(question, k)` returns chunk text, source filename, one-based page number for display, chunk ID, and L2 distance. Lower L2 distance means a closer embedding-space match. The verified retrieval checks confirm that the same query yields the same top-k order and distances across consecutive runs, while different questions produce different chunk sets.

## Query routing and decomposition

`query_router.py` classifies queries as `single_fact`, `multi_part`, or `summarization`. When supplied with an LLM, it requests JSON containing a query type and sub-query list, validates the response, and falls back to deterministic rules if the response is empty, malformed, or invalid.

`main.py` runs retrieval for every sub-query independently, records each query's retrieval trace, merges the selected chunks, and removes duplicates before building the context. Check `routing_method` in the output: the latest integrated reranking test used `rules` because the LLM routing response failed validation, while an earlier integrated run successfully used `llm`.

## LCEL chain and reranking integration

`main.py` constructs an LCEL pipeline that:

1. Routes the original question and decomposes multi-part questions.
2. Retrieves candidate chunks from FAISS separately for each sub-query.
3. Reranks candidates with `cross-encoder/ms-marco-MiniLM-L-6-v2`.
4. Selects the highest-ranked chunks, merges selected results, and removes duplicate chunks.
5. Sends the selected context to the OpenRouter chat model through the prompt and output parser.
6. Returns the answer, query type, routing method, sub-queries, source metadata, and ranking trace.

The integrated test question was: “How many casual leaves are allocated on a half-yearly basis, and when are the two half-yearly cycles?” The latest run returned 4 casual leaves per half-year, covering January–June and July–December, with a citation to `Veer_IT_Solutions_India_Leave_Policy.pdf`, page 3, chunk 109. It printed pre-rerank and post-rerank rankings for both sub-queries; `Top result changed: True` appeared for each. The final selected context was taken from the post-rerank ordering.

The router's LLM call fell back to rules for this particular run, but answer generation succeeded. The chain has also previously passed an out-of-scope stock-ticker test by stating that the policy documents did not contain that information.

## Reranking evidence

The reranker obtains ten candidates per test question from FAISS and scores each question–chunk pair with the cross-encoder. The standalone experiment tests six questions and writes the ranking data, snippets, scores, and top-result-change flag to:

```text
evaluation/reranking_comparison.json
```

In the standalone casual-leave example, FAISS ranked chunk 102 on page 1 first; the cross-encoder ranked chunk 109 on page 3 first. The integrated chain also recorded changed top results for both sub-queries in its latest successful run.

A genuine terminal screenshot has not yet been captured into the repository. The intended path is `evaluation/screenshots/reranking-comparison.png`; capture it locally from the actual terminal output before embedding it in the root README.

## Remaining backend work

1. Improve resilience and observability for LLM JSON routing; verify both LLM and rule-based paths with tests.
2. Capture the actual terminal screenshot of the integrated before/after ranking evidence.
3. Run and log comprehensive RAGAS evaluation metrics for faithfulness, answer relevancy, and context precision.
4. Deploy the backend and frontend services to production hosting.
