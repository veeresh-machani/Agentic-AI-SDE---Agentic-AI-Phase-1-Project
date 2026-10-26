import json
from datetime import datetime, timezone
from functools import lru_cache
from pathlib import Path

import torch
from sentence_transformers import CrossEncoder

torch.set_grad_enabled(False)

from backend.app.retrieval.query_index import load_vector_store


RERANKER_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"
CANDIDATE_K = 10
DISPLAY_K = 5

PROJECT_ROOT = Path(__file__).resolve().parents[3]
EVIDENCE_PATH = PROJECT_ROOT / "evaluation" / "reranking_comparison.json"

TEST_QUESTIONS = [
    "What is the process for applying for casual leave?",
    "How many casual leaves are allocated on a half-yearly basis?",
    "What are the rules for carrying forward privileged leave?",
    "What is the eligibility for paternity leave?",
    "How is employee performance appraisal conducted?",
    "What are the rules for compensatory off?",
]


@lru_cache(maxsize=1)
def load_reranker() -> CrossEncoder:
    """Load the cross-encoder once per process."""
    return CrossEncoder(RERANKER_MODEL, device="cpu")


def identify_chunk(document) -> tuple:
    """Identify a chunk consistently across both rankings."""
    metadata = document.metadata
    return (
        str(metadata.get("source", "Unknown")),
        metadata.get("page"),
        metadata.get("chunk_id"),
    )


def describe_chunk(document) -> dict:
    """Return source metadata and a text excerpt for evidence."""
    metadata = document.metadata

    source = Path(str(metadata.get("source", "Unknown"))).name

    return {
        "source": source,
        "page": int(metadata.get("page", 0)) + 1,
        "chunk_id": metadata.get("chunk_id"),
        "snippet": document.page_content[:300],
    }


def rerank_documents(question, documents):
    """Rank the FAISS candidates by cross-encoder score."""
    if not documents:
        return []

    pairs = [(question, document.page_content) for document in documents]

    with torch.inference_mode():
        scores = load_reranker().predict(pairs)

    ranked = sorted(
        zip(documents, scores),
        key=lambda item: float(item[1]),
        reverse=True,
    )

    return [(document, float(score)) for document, score in ranked]


def evaluate_query(vector_store, question):
    """Compare FAISS ordering with cross-encoder ordering."""

    faiss_results = vector_store.similarity_search_with_score(
        question,
        k=CANDIDATE_K,
    )

    candidates = [document for document, _ in faiss_results]

    reranked_results = rerank_documents(
        question,
        candidates,
    )

    before = []
    for rank, (document, distance) in enumerate(faiss_results, start=1):
        item = describe_chunk(document)
        item.update(
            {
                "rank": rank,
                "l2_distance": float(distance),
            }
        )
        before.append(item)

    after = []
    for rank, (document, score) in enumerate(reranked_results, start=1):
        item = describe_chunk(document)
        item.update(
            {
                "rank": rank,
                "cross_encoder_score": score,
            }
        )
        after.append(item)

    before_top = identify_chunk(faiss_results[0][0]) if faiss_results else None
    after_top = identify_chunk(reranked_results[0][0]) if reranked_results else None

    return {
        "question": question,
        "candidate_count": len(candidates),
        "top_result_changed": before_top != after_top,
        "before_reranking": before,
        "after_reranking": after,
    }


def main():
    vector_store = load_vector_store()

    comparisons = []

    for question in TEST_QUESTIONS:
        print(f"\nTesting: {question}")
        result = evaluate_query(vector_store, question)
        comparisons.append(result)

        print(
            "Top result changed:",
            result["top_result_changed"],
        )

    # Persist every tested query and both complete rankings.
    EVIDENCE_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    evidence = {
        "model": RERANKER_MODEL,
        "candidate_k": CANDIDATE_K,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "comparisons": comparisons,
    }

    EVIDENCE_PATH.write_text(
        json.dumps(evidence, indent=2),
        encoding="utf-8",
    )

    # Display a concrete before/after example when one exists.
    changed = next(
        (item for item in comparisons if item["top_result_changed"]),
        None,
    )

    print("\n" + "=" * 70)
    print("RERANKING EVIDENCE")
    print("=" * 70)

    if changed:
        print("\nQuery with a changed top result:")
        print(changed["question"])

        print("\nBEFORE RERANKING — FAISS ORDER")
        for item in changed["before_reranking"][:DISPLAY_K]:
            print(
                f"{item['rank']}. {item['source']} | "
                f"Page {item['page']} | "
                f"Chunk {item['chunk_id']} | "
                f"L2 distance: {item['l2_distance']:.4f}"
            )
            print(f"   {item['snippet'][:180]}")

        print("\nAFTER RERANKING — CROSS-ENCODER ORDER")
        for item in changed["after_reranking"][:DISPLAY_K]:
            print(
                f"{item['rank']}. {item['source']} | "
                f"Page {item['page']} | "
                f"Chunk {item['chunk_id']} | "
                f"Score: {item['cross_encoder_score']:.4f}"
            )
            print(f"   {item['snippet'][:180]}")

        print("\nTOP RESULT CHANGED: YES")
    else:
        print(
            "No tested question changed its top-ranked chunk. "
            "Do not claim the ranking-change checklist passed yet."
        )

    print("\nQueries tested:", len(comparisons))
    print(
        "Queries with changed top result:",
        sum(item["top_result_changed"] for item in comparisons),
    )
    print("Evidence saved to:", EVIDENCE_PATH)


if __name__ == "__main__":
    main()
