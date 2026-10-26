import os
from pathlib import Path
from dotenv import load_dotenv
from langchain_core.documents import Document
from langchain_openai import OpenAIEmbeddings

PROJECT_ROOT = Path(__file__).resolve().parents[3]
load_dotenv(PROJECT_ROOT / ".env")

MODEL_NAME = "sentence-transformers/all-minilm-l6-v2"


def create_embeddings() -> OpenAIEmbeddings:
    """Create lightweight cloud embeddings via OpenRouter (0 MB local RAM)."""
    api_key = os.getenv("OPENROUTER_API_KEY")
    if not api_key:
        raise RuntimeError("OPENROUTER_API_KEY is missing from environment.")

    return OpenAIEmbeddings(
        model=MODEL_NAME,
        openai_api_key=api_key,
        openai_api_base="https://openrouter.ai/api/v1",
        check_embedding_ctx_length=False,
    )


def embed_documents(
    documents: list[Document],
) -> tuple[OpenAIEmbeddings, list[list[float]]]:
    """Generate embeddings for a list of LangChain Documents."""
    embedding_model = create_embeddings()
    texts = [document.page_content for document in documents]
    vectors = embedding_model.embed_documents(texts)
    return embedding_model, vectors


if __name__ == "__main__":
    from backend.app.ingestion.chunker import chunk_documents
    from backend.app.ingestion.pdf_loader import load_all_pdfs

    documents = load_all_pdfs()
    chunks = chunk_documents(documents)

    print("\n" + "=" * 60)
    print("CREATING EMBEDDINGS (CLOUD API)")
    print("=" * 60)
    print(f"Chunks: {len(chunks)}")
    print(f"Model : {MODEL_NAME}")

    embedding_model, vectors = embed_documents(chunks[:5])

    print("\n" + "=" * 60)
    print("EMBEDDING COMPLETE")
    print("=" * 60)
    print(f"Chunks embedded : {len(vectors)}")
    print(f"Vector dimension: {len(vectors[0])}")
    print("\nFirst vector preview:")
    print(vectors[0][:10])
