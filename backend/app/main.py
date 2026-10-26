import logging
import os
from collections import OrderedDict
from functools import lru_cache
from pathlib import Path
from uuid import uuid4

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from backend.app.ingestion.upload_processor import (
    ProcessedUpload,
    process_pdf_bytes,
)
from backend.app.rag.main import build_rag_chain, create_llm


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

MAX_UPLOAD_BYTES = 10 * 1024 * 1024  # 10 MB
MAX_ACTIVE_UPLOADS = 5

# In-memory upload registry for this demo.
# A server restart clears these uploads.
UPLOADS: OrderedDict[str, ProcessedUpload] = OrderedDict()


app = FastAPI(
    title="HR Policy Bot API",
    description="Upload a PDF and ask questions grounded in its contents.",
    version="1.1.0",
)

default_origins = [
    "http://localhost:3000",
    "http://127.0.0.1:3000",
]
configured_origins = [
    origin.strip()
    for origin in os.getenv("FRONTEND_ORIGINS", "").split(",")
    if origin.strip()
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=list(dict.fromkeys(default_origins + configured_origins)),
    allow_credentials=False,
    allow_methods=["GET", "POST"],
    allow_headers=["Content-Type"],
)


class UploadResponse(BaseModel):
    upload_id: str
    filename: str
    page_count: int
    chunk_count: int


class QuestionRequest(BaseModel):
    question: str = Field(
        min_length=1,
        max_length=2000,
        description="A question about the HR policies or uploaded PDF.",
    )
    upload_id: str | None = Field(
        default="default",
        max_length=64,
        description="The ID returned by POST /api/upload, or 'default' for pre-loaded policies.",
    )


class SourceItem(BaseModel):
    source: str
    page: int
    chunk_id: int | None = None
    snippet: str


class QuestionResponse(BaseModel):
    question: str
    answer: str
    query_type: str
    routing_method: str
    sources: list[SourceItem]


@lru_cache(maxsize=1)
def get_llm():
    """Initialize the language model once per server process."""
    return create_llm()


@app.get("/health")
def health():
    return {"status": "ok"}


@app.get("/api/default-document", response_model=UploadResponse)
def get_default_document():
    """Return metadata for the default pre-loaded HR Policy Knowledge Base."""
    return UploadResponse(
        upload_id="default",
        filename="Veer IT Solutions HR Policy Knowledge Base",
        page_count=39,
        chunk_count=124,
    )


@app.post("/api/upload", response_model=UploadResponse, status_code=201)
def upload_pdf(file: UploadFile = File(...)):
    """Read, chunk, embed, and index a user-uploaded PDF."""
    filename = Path(file.filename or "").name

    if not filename or Path(filename).suffix.lower() != ".pdf":
        raise HTTPException(
            status_code=415,
            detail="Please upload a file with a .pdf extension.",
        )

    file_bytes = file.file.read(MAX_UPLOAD_BYTES + 1)

    if not file_bytes:
        raise HTTPException(status_code=422, detail="The uploaded file is empty.")

    if len(file_bytes) > MAX_UPLOAD_BYTES:
        raise HTTPException(
            status_code=413,
            detail="The PDF is too large. The maximum upload size is 10 MB.",
        )

    if b"%PDF-" not in file_bytes[:1024]:
        raise HTTPException(
            status_code=422,
            detail="The uploaded file does not appear to be a valid PDF.",
        )

    try:
        processed = process_pdf_bytes(file_bytes, filename)
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    except Exception as exc:
        logger.exception("Failed to process uploaded PDF")
        raise HTTPException(
            status_code=500,
            detail="The PDF could not be indexed. Please try another PDF.",
        ) from exc

    upload_id = uuid4().hex

    # Retain only a small number of indexes in memory.
    while len(UPLOADS) >= MAX_ACTIVE_UPLOADS:
        UPLOADS.popitem(last=False)

    UPLOADS[upload_id] = processed

    logger.info(
        "Indexed upload %s: %s pages, %s chunks",
        filename,
        processed.page_count,
        processed.chunk_count,
    )

    return UploadResponse(
        upload_id=upload_id,
        filename=processed.filename,
        page_count=processed.page_count,
        chunk_count=processed.chunk_count,
    )


@app.post("/api/query", response_model=QuestionResponse)
def query_uploaded_pdf(request: QuestionRequest):
    """Answer a question using either an uploaded PDF or the default HR policy knowledge base."""
    question = request.question.strip()

    if not question:
        raise HTTPException(
            status_code=422,
            detail="Question cannot be empty.",
        )

    vector_store = None
    upload_id = (request.upload_id or "").strip()

    # If a specific uploaded session ID is provided and is not the default
    if upload_id and upload_id != "default":
        uploaded = UPLOADS.get(upload_id)
        if uploaded is not None:
            UPLOADS.move_to_end(upload_id)
            vector_store = uploaded.vector_store
        else:
            logger.info(
                "Upload %s not found in memory; falling back to default HR policy knowledge base",
                upload_id,
            )
            # Falls back to default vector_store = None, which uses load_vector_store()

    try:
        # When vector_store is None, build_rag_chain automatically loads the default FAISS store
        chain = build_rag_chain(
            get_llm(),
            k=4,
            vector_store=vector_store,
        )
        result = chain.invoke({"question": question})

        return QuestionResponse(
            question=question,
            answer=result["answer"],
            query_type=result["query_type"],
            routing_method=result["routing_method"],
            sources=result["sources"],
        )

    except Exception as exc:
        logger.exception("HR policy query failed")
        raise HTTPException(
            status_code=502,
            detail=(
                "The question could not be answered. "
                "The language-model provider may be unavailable or rate-limited."
            ),
        ) from exc
