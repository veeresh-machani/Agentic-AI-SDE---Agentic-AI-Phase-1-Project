from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from tempfile import TemporaryDirectory

from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import FAISS
from langchain_text_splitters import RecursiveCharacterTextSplitter

from backend.app.retrieval.embeddings import create_embeddings


@dataclass
class ProcessedUpload:
    filename: str
    page_count: int
    chunk_count: int
    vector_store: FAISS


@lru_cache(maxsize=1)
def get_embedding_model():
    """Load and reuse the local embedding model."""
    return create_embeddings()


def process_pdf_bytes(file_bytes: bytes, filename: str) -> ProcessedUpload:
    """Extract, chunk, embed, and index one uploaded PDF."""
    safe_filename = Path(filename).name

    if not file_bytes:
        raise ValueError("The uploaded PDF is empty.")

    with TemporaryDirectory() as temp_dir:
        pdf_path = Path(temp_dir) / "uploaded.pdf"
        pdf_path.write_bytes(file_bytes)

        try:
            pages = PyPDFLoader(str(pdf_path)).load()
        except Exception as exc:
            raise ValueError(
                "This file could not be read as a PDF."
            ) from exc

    if not pages:
        raise ValueError("No pages were found in this PDF.")

    if len(pages) > 150:
        raise ValueError(
            "This demo accepts PDFs with at most 150 pages."
        )

    if not any(page.page_content.strip() for page in pages):
        raise ValueError(
            "No selectable text was found. Scanned image-only PDFs "
            "are not supported by this version."
        )

    # Replace temporary-path metadata with the uploaded filename.
    for page in pages:
        page.metadata["source"] = safe_filename

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=800,
        chunk_overlap=120,
    )
    chunks = splitter.split_documents(pages)

    chunks = [
        chunk for chunk in chunks
        if chunk.page_content.strip()
    ]

    if not chunks:
        raise ValueError("No text chunks could be created from this PDF.")

    for chunk_id, chunk in enumerate(chunks):
        chunk.metadata["chunk_id"] = chunk_id

    vector_store = FAISS.from_documents(
        documents=chunks,
        embedding=get_embedding_model(),
    )

    return ProcessedUpload(
        filename=safe_filename,
        page_count=len(pages),
        chunk_count=len(chunks),
        vector_store=vector_store,
    ) 