from langchain_core.documents import Document
from langchain_huggingface import HuggingFaceEmbeddings

MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"


def create_embeddings() -> HuggingFaceEmbeddings:
    """
    Create the embedding model used by the HR Policy Bot.
    """

    embeddings = HuggingFaceEmbeddings(
        model_name=MODEL_NAME,
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True},
    )

    return embeddings


def embed_documents(
    documents: list[Document],
) -> tuple[HuggingFaceEmbeddings, list[list[float]]]:
    """
    Generate embeddings for a list of LangChain Documents.
    """

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
    print("CREATING EMBEDDINGS")
    print("=" * 60)

    print(f"Chunks: {len(chunks)}")
    print(f"Model : {MODEL_NAME}")

    embedding_model, vectors = embed_documents(chunks)

    print("\n" + "=" * 60)
    print("EMBEDDING COMPLETE")
    print("=" * 60)

    print(f"Chunks embedded : {len(vectors)}")
    print(f"Vector dimension: {len(vectors[0])}")

    print("\nFirst vector preview:")
    print(vectors[0][:10])
