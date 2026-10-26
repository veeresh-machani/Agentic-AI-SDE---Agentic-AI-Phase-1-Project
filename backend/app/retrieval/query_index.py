from functools import lru_cache
from pathlib import Path

from langchain_community.vectorstores import FAISS

from backend.app.retrieval.embeddings import create_embeddings

# Saved FAISS index location
INDEX_DIR = Path(__file__).resolve().parents[2] / "data" / "index" / "hr_policy_faiss"


@lru_cache(maxsize=1)
def load_vector_store() -> FAISS:
    """
    Load and cache the saved FAISS index.
    """

    if not (INDEX_DIR / "index.faiss").exists():
        raise FileNotFoundError(
            f"FAISS index not found at {INDEX_DIR}. "
            "Run python -m backend.app.retrieval.faiss_store first."
        )

    embedding_model = create_embeddings()

    return FAISS.load_local(
        str(INDEX_DIR),
        embedding_model,
        allow_dangerous_deserialization=True,
    )


def query_index(question: str, k: int = 4) -> list[dict]:
    """
    Retrieve the top-k chunks relevant to a question.

    Returns each chunk's text, source, page number,
    chunk ID, and L2 distance.

    Lower L2 distance indicates a closer match.
    """

    if not question.strip():
        raise ValueError("Question cannot be empty.")

    if k < 1:
        raise ValueError("k must be at least 1.")

    vector_store = load_vector_store()

    results = vector_store.similarity_search_with_score(
        question,
        k=k,
    )

    retrieved_chunks = []

    for document, distance in results:
        source = document.metadata.get("source", "Unknown")
        page = document.metadata.get("page", 0)

        retrieved_chunks.append(
            {
                "chunk_id": document.metadata.get("chunk_id"),
                "source": Path(str(source)).name,
                "page": int(page) + 1,
                "content": document.page_content,
                "distance": float(distance),
            }
        )

    return retrieved_chunks


if __name__ == "__main__":
    question = "What is the process for applying for leave?"

    results = query_index(question, k=4)

    print("\nQUESTION:", question)
    print("=" * 60)

    for rank, result in enumerate(results, start=1):
        print(f"\nResult {rank}")
        print(f"Source   : {result['source']}")
        print(f"Page     : {result['page']}")
        print(f"Chunk ID : {result['chunk_id']}")
        print(f"L2 distance: {result['distance']:.4f}")
        print(f"Content  : {result['content'][:400]}")
