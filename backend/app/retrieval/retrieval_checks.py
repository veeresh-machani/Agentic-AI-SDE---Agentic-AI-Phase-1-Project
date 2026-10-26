from math import isclose

from backend.app.retrieval.query_index import query_index


def chunk_ids(results):
    return [(item["source"], item["page"], item["chunk_id"]) for item in results]


def main():
    leave_question = "What is the process for applying for leave?"
    appraisal_question = "How is employee performance appraisal conducted?"

    # Test 1: Repeat the same question.
    first = query_index(leave_question, k=4)
    repeated = query_index(leave_question, k=4)

    same_chunks = chunk_ids(first) == chunk_ids(repeated)

    same_scores = all(
        isclose(a["distance"], b["distance"], rel_tol=1e-6, abs_tol=1e-6)
        for a, b in zip(first, repeated)
    )

    print("=" * 60)
    print("TEST 1: SAME QUESTION TWICE")
    print("=" * 60)
    print("Chunk ordering identical:", same_chunks)
    print("Distances identical:", same_scores)
    print("RESULT:", "PASS" if same_chunks and same_scores else "FAIL")

    # Test 2: Compare different HR questions.
    appraisal_results = query_index(appraisal_question, k=4)

    leave_ids = set(chunk_ids(first))
    appraisal_ids = set(chunk_ids(appraisal_results))

    different_chunks = leave_ids != appraisal_ids

    print("\n" + "=" * 60)
    print("TEST 2: DIFFERENT QUESTIONS")
    print("=" * 60)
    print("Leave question:", leave_question)
    print("Appraisal question:", appraisal_question)
    print("Leave result chunk IDs:", chunk_ids(first))
    print("Appraisal result chunk IDs:", chunk_ids(appraisal_results))
    print("Different chunk sets:", different_chunks)
    print("RESULT:", "PASS" if different_chunks else "FAIL")


if __name__ == "__main__":
    main()
