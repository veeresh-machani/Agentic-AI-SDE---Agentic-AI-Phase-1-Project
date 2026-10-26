from pathlib import Path

from langchain_community.vectorstores import FAISS

from backend.app.ingestion.chunker import chunk_documents
from backend.app.ingestion.pdf_loader import load_all_pdfs
from backend.app.retrieval.embeddings import create_embeddings

# Resolve paths relative to this file, not the terminal's directory.
PROJECT_ROOT = Path(__file__).resolve().parents[3]

DOCUMENTS_DIR = PROJECT_ROOT / "data" / "documents"

INDEX_DIR = PROJECT_ROOT / "backend" / "data" / "index" / "hr_policy_faiss"


def build_faiss_index() -> None:
    """Load HR policies, create chunks, and save a FAISS index."""

    # 1. Load PDFs page by page.
    documents = load_all_pdfs(str(DOCUMENTS_DIR))

    # 2. Split pages into searchable chunks.
    chunks = chunk_documents(documents)

    # 3. Initialize the embedding model.
    embedding_model = create_embeddings()

    # 4. Create the FAISS vector index.
    vector_store = FAISS.from_documents(
        documents=chunks,
        embedding=embedding_model,
    )

    # 5. Save the index to disk.
    INDEX_DIR.mkdir(parents=True, exist_ok=True)
    vector_store.save_local(str(INDEX_DIR))

    print("\n" + "=" * 50)
    print("FAISS INDEX CREATED SUCCESSFULLY")
    print("=" * 50)
    print(f"Pages loaded : {len(documents)}")
    print(f"Chunks indexed: {len(chunks)}")
    print(f"Index saved to: {INDEX_DIR}")


if __name__ == "__main__":
    build_faiss_index()
