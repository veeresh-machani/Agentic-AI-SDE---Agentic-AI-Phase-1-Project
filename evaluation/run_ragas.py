import argparse
import json
import os
from datetime import datetime, timezone
from pathlib import Path

from dotenv import load_dotenv
from openai import AsyncOpenAI
from ragas.embeddings import HuggingFaceEmbeddings
from ragas.llms import llm_factory
from ragas.metrics.collections import (
    AnswerRelevancy,
    ContextPrecision,
    Faithfulness,
)

from backend.app.rag.main import build_rag_chain, create_llm

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATASET_PATH = PROJECT_ROOT / "evaluation" / "hr_eval_dataset.json"
RESULTS_PATH = PROJECT_ROOT / "evaluation" / "ragas_results.jsonl"


def create_evaluator():
    """Create RAGAS evaluator LLM and local evaluator embeddings."""

    api_key = os.getenv("OPENROUTER_API_KEY")
    if not api_key:
        raise RuntimeError("OPENROUTER_API_KEY is missing from the root .env file.")

    model = os.getenv(
        "RAGAS_MODEL",
        os.getenv("OPENROUTER_MODEL", "openrouter/free"),
    ).strip()

    client = AsyncOpenAI(
        api_key=api_key,
        base_url="https://openrouter.ai/api/v1",
        timeout=90,
        max_retries=1,
    )

    evaluator_llm = llm_factory(
        model=model,
        provider="openai",
        client=client,
    )

    evaluator_embeddings = HuggingFaceEmbeddings(
        model="sentence-transformers/all-MiniLM-L6-v2",
        device="cpu",
        normalize_embeddings=True,
    )

    return evaluator_llm, evaluator_embeddings


def score_value(metric, **kwargs) -> float:
    """Return a numeric score from a RAGAS collection metric."""
    result = metric.score(**kwargs)
    value = getattr(result, "value", result)
    return float(value)


def main(limit: int | None = None):
    load_dotenv(PROJECT_ROOT / ".env")

    if not DATASET_PATH.exists():
        raise FileNotFoundError(f"Evaluation dataset not found: {DATASET_PATH}")

    cases = json.loads(DATASET_PATH.read_text(encoding="utf-8"))

    if limit is not None:
        cases = cases[:limit]

    if not cases:
        raise ValueError("The evaluation dataset is empty.")

    evaluator_llm, evaluator_embeddings = create_evaluator()

    metrics = {
        "faithfulness": Faithfulness(llm=evaluator_llm),
        "answer_relevancy": AnswerRelevancy(
            llm=evaluator_llm,
            embeddings=evaluator_embeddings,
        ),
        "context_precision": ContextPrecision(llm=evaluator_llm),
    }

    # Reuse the production chain, including routing and reranking.
    chain = build_rag_chain(create_llm(), k=4)

    RESULTS_PATH.parent.mkdir(parents=True, exist_ok=True)

    successful_scores: dict[str, list[float]] = {name: [] for name in metrics}

    with RESULTS_PATH.open("w", encoding="utf-8") as output_file:
        for index, case in enumerate(cases, start=1):
            question = case["question"]
            reference = case["reference"]

            record = {
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "id": case["id"],
                "question": question,
                "reference": reference,
                "status": "started",
                "answer_preview": None,
                "metrics": {
                    "faithfulness": None,
                    "answer_relevancy": None,
                    "context_precision": None,
                },
                "metric_errors": {},
            }

            print(f"\n[{index}/{len(cases)}] {question}")

            try:
                result = chain.invoke({"question": question})
                answer = result["answer"]
                contexts = result.get("retrieved_contexts", [])

                if not contexts:
                    raise ValueError(
                        "The RAG chain returned no retrieved contexts. "
                        "Check the retrieved_contexts field in main.py."
                    )

                record["answer_preview"] = answer[:400]
                record["context_count"] = len(contexts)
                record["sources"] = result.get("sources", [])
                record["routing_method"] = result.get("routing_method")
                record["query_type"] = result.get("query_type")

            except Exception as exc:
                record["status"] = "generation_failed"
                record["generation_error"] = f"{type(exc).__name__}: {exc}"
                output_file.write(json.dumps(record) + "\n")
                output_file.flush()
                print("Generation failed:", record["generation_error"])
                continue

            metric_inputs = {
                "faithfulness": {
                    "user_input": question,
                    "response": answer,
                    "retrieved_contexts": contexts,
                },
                "answer_relevancy": {
                    "user_input": question,
                    "response": answer,
                },
                "context_precision": {
                    "user_input": question,
                    "reference": reference,
                    "retrieved_contexts": contexts,
                },
            }

            for name, metric in metrics.items():
                try:
                    score = score_value(
                        metric,
                        **metric_inputs[name],
                    )
                    record["metrics"][name] = score
                    successful_scores[name].append(score)
                    print(f"  {name}: {score:.4f}")

                except Exception as exc:
                    record["metric_errors"][name] = f"{type(exc).__name__}: {exc}"
                    print(f"  {name}: FAILED ({type(exc).__name__})")

            record["status"] = (
                "completed"
                if all(value is not None for value in record["metrics"].values())
                else "partial"
            )

            output_file.write(json.dumps(record, ensure_ascii=False) + "\n")
            output_file.flush()

    print("\n" + "=" * 60)
    print("RAGAS EVALUATION SUMMARY")
    print("=" * 60)

    for name, scores in successful_scores.items():
        if scores:
            average = sum(scores) / len(scores)
            print(f"{name}: {average:.4f} ({len(scores)} successful scores)")
        else:
            print(f"{name}: no successful scores")

    print(f"\nJSONL results saved to: {RESULTS_PATH}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Run only the first N evaluation cases (useful for a smoke test).",
    )
    args = parser.parse_args()
    main(limit=args.limit)
