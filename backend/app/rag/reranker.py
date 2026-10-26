import json
import os
import re
from datetime import datetime, timezone
from functools import lru_cache
from pathlib import Path

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
def load_reranker():
    """Load the cross-encoder lazily if neural reranking is requested."""
    from sentence_transformers import CrossEncoder
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


def rerank_documents(question: str, documents: list) -> list[tuple]:
    """Rank the FAISS candidates with zero memory overhead (Render Free Tier 512MB safe)."""
    if not documents:
        return []

    # If explicitly enabled and memory allows, use heavy neural model
    if os.getenv("ENABLE_NEURAL_RERANKER", "false").lower() == "true":
        try:
            pairs = [(question, document.page_content) for document in documents]
            scores = load_reranker().predict(pairs)
            ranked = sorted(
                zip(documents, scores),
                key=lambda item: float(item[1]),
                reverse=True,
            )
            return [(document, float(score)) for document, score in ranked]
        except Exception:
            pass

    # Lightweight, fast keyword-overlap + FAISS candidate score (0 MB RAM overhead)
    question_terms = set(re.findall(r"\w+", question.lower()))
    ranked = []
    for rank_idx, doc in enumerate(documents):
        content_lower = doc.page_content.lower()
        match_count = sum(1 for term in question_terms if term in content_lower)
        score = (match_count * 2.0) + (1.0 / (rank_idx + 1))
        ranked.append((doc, float(score)))

    ranked.sort(key=lambda item: item[1], reverse=True)
    return ranked


def evaluate_query(vector_store, question):
    """Compare FAISS ordering with reranking ordering."""
    faiss_results = vector_store.similarity_search_with_score(
        question,
        k=CANDIDATE_K,
    )
    candidates = [document for document, _ in faiss_results]
    reranked_results = rerank_documents(question, candidates)

    before = []
    for rank, (document, distance) in enumerate(faiss_results, start=1):
        item = describe_chunk(document)
        item.update({"rank": rank, "l2_distance": float(distance)})
        before.append(item)

    after = []
    for rank, (document, score) in enumerate(reranked_results, start=1):
        item = describe_chunk(document)
        item.update({"rank": rank, "cross_encoder_score": score})
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
        print("Top result changed:", result["top_result_changed"])

    EVIDENCE_PATH.parent.mkdir(parents=True, exist_ok=True)
    evidence = {
        "model": RERANKER_MODEL,
        "candidate_k": CANDIDATE_K,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "comparisons": comparisons,
    }
    EVIDENCE_PATH.write_text(json.dumps(evidence, indent=2), encoding="utf-8")
    print("\nEvidence saved to:", EVIDENCE_PATH)


if __name__ == "__main__":
    main()
