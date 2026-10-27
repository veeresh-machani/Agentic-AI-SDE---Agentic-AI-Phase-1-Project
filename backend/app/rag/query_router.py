import json
import re
from dataclasses import dataclass
from typing import Literal

from langchain_core.messages import HumanMessage, SystemMessage

QueryType = Literal[
    "single_fact",
    "multi_part",
    "summarization",
]

RoutingMethod = Literal["llm", "rules"]


@dataclass(frozen=True)
class QueryPlan:
    query_type: QueryType
    sub_queries: list[str]
    routing_method: RoutingMethod


SUMMARY_PATTERN = re.compile(
    r"\b(summarize|summarise|summary|overview|"
    r"key points|main points|outline)\b",
    re.IGNORECASE,
)

COMPARISON_PATTERN = re.compile(
    r"\b(?:difference between|compare|contrast)\s+"
    r"(.+?)\s+(?:and|with|to|versus|vs\.?)\s+"
    r"(.+?)(?:[?.!]|$)",
    re.IGNORECASE,
)

MULTI_PART_SPLIT_PATTERN = re.compile(
    r"\s+(?:and|also|additionally|plus|as well as)\s+"
    r"(?=(?:what|how|when|where|why|who|which|is|are|"
    r"does|do|can|could|would|should|explain|describe|"
    r"compare|list|tell me|give me)\b)",
    re.IGNORECASE,
)


def clean_question(text: str) -> str:
    """Normalize the whitespace and punctuation of a sub-question."""

    text = re.sub(r"\s+", " ", text).strip(" \t\n,;:?!.")

    if text:
        text = text[0].upper() + text[1:]
        if not text.endswith("?"):
            text += "?"

    return text


def _extract_json(text: str) -> dict:
    """Parse JSON even if the model surrounds it with think tags or code fences."""
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL).strip()
    text = re.sub(
        r"```(?:json)?\s*([\s\S]*?)\s*```",
        r"\1",
        text,
        flags=re.IGNORECASE,
    ).strip()

    # Allow harmless text before or after a JSON object.
    start = text.find("{")
    end = text.rfind("}")

    if start == -1 or end < start:
        raise ValueError("The router did not return a JSON object.")

    result = json.loads(text[start : end + 1])

    if not isinstance(result, dict):
        raise ValueError("The router JSON must be an object.")

    return result


def _route_with_llm(question: str, llm) -> QueryPlan:
    """Ask the LLM to classify and decompose the question."""

    system_prompt = """
You are a query router for an HR Policy RAG system.

Classify the user's question into exactly one category:

1. single_fact:
   One fact or one closely related information request.

2. multi_part:
   Two or more independently answerable requests, or a
   comparison that needs separate evidence for each subject.

3. summarization:
   A request to summarize, outline, or provide an overview
   of a policy or a group of related policy details.

For multi_part questions:
- Split the original request into focused sub-questions.
- Preserve every part of the user's information request.
- For comparisons, include sub-questions that retrieve evidence
  for both subjects. A separate comparison sub-question is optional.
- Return between 2 and 5 sub-questions.

For single_fact and summarization, return the original question
as the only sub-question.

Return only valid JSON. Do not include Markdown or explanations.
Use exactly this schema:

{
  "query_type": "single_fact",
  "sub_queries": ["Original question?"]
}

The query_type value must be one of:
single_fact, multi_part, summarization.
"""

    response = llm.invoke(
        [
            SystemMessage(content=system_prompt),
            HumanMessage(content=f"Question:\n{question}"),
        ]
    )

    content = response.content

    if isinstance(content, list):
        text_parts = []
        for item in content:
            if isinstance(item, dict):
                if item.get("type") == "text":
                    text_parts.append(item.get("text", ""))
            elif isinstance(item, str):
                text_parts.append(item)
        content = "\n".join(text_parts)

    if not isinstance(content, str) or not content.strip():
        if hasattr(response, "additional_kwargs"):
            reasoning = response.additional_kwargs.get("reasoning_content", "")
            if isinstance(reasoning, str) and reasoning.strip():
                content = reasoning.strip()

    if not isinstance(content, str) or not content.strip():
        raise ValueError("The router returned an empty response.")

    data = _extract_json(content)

    query_type = data.get("query_type")
    if query_type not in {
        "single_fact",
        "multi_part",
        "summarization",
    }:
        raise ValueError("The router returned an unknown query type.")

    raw_queries = data.get("sub_queries")
    if not isinstance(raw_queries, list):
        raise ValueError("sub_queries must be a JSON list.")

    sub_queries = []
    seen = set()

    for item in raw_queries:
        if not isinstance(item, str):
            continue

        cleaned = clean_question(item)

        if cleaned and cleaned.casefold() not in seen:
            seen.add(cleaned.casefold())
            sub_queries.append(cleaned)

    if query_type == "multi_part":
        if len(sub_queries) < 2:
            raise ValueError("A multi-part route requires at least two sub-questions.")
        sub_queries = sub_queries[:5]
    else:
        # Keep the original question intact for these categories.
        sub_queries = [question]

    return QueryPlan(
        query_type=query_type,
        sub_queries=sub_queries,
        routing_method="llm",
    )


def _route_with_rules(question: str) -> QueryPlan:
    """Deterministic fallback if LLM routing fails."""

    if SUMMARY_PATTERN.search(question):
        return QueryPlan(
            query_type="summarization",
            sub_queries=[question],
            routing_method="rules",
        )

    comparison = COMPARISON_PATTERN.search(question)

    if comparison:
        left = comparison.group(1).strip(" ,;:.?!")
        right = comparison.group(2).strip(" ,;:.?!")

        if left and right:
            return QueryPlan(
                query_type="multi_part",
                sub_queries=[
                    clean_question(f"What does the HR policy say about {left}"),
                    clean_question(f"What does the HR policy say about {right}"),
                    clean_question(
                        f"What is the difference between {left} and {right}"
                    ),
                ],
                routing_method="rules",
            )

    parts = MULTI_PART_SPLIT_PATTERN.split(question)
    sub_queries = [clean_question(part) for part in parts if part.strip(" \t\n,;:.?!")]

    if len(sub_queries) > 1:
        return QueryPlan(
            query_type="multi_part",
            sub_queries=sub_queries[:5],
            routing_method="rules",
        )

    return QueryPlan(
        query_type="single_fact",
        sub_queries=[question],
        routing_method="rules",
    )


def route_query(question: str, llm=None) -> QueryPlan:
    """
    Prefer LLM-based JSON routing when an LLM is supplied.

    If routing fails or returns invalid JSON, use deterministic
    rules instead of stopping the entire pipeline.
    """

    question = re.sub(r"\s+", " ", question).strip()

    if not question:
        raise ValueError("Question cannot be empty.")

    if llm is not None:
        try:
            return _route_with_llm(question, llm)
        except Exception as exc:
            print(
                "Warning: LLM routing failed; using rule-based "
                f"fallback ({type(exc).__name__})."
            )

    return _route_with_rules(question)


if __name__ == "__main__":
    examples = [
        "How many casual leaves are allocated?",
        (
            "How many casual leaves are allocated, and "
            "when are the two half-yearly cycles?"
        ),
        "Compare casual leave and privileged leave.",
        "Summarize the main leave policies.",
    ]

    # These examples exercise the deterministic fallback.
    for question in examples:
        plan = route_query(question)

        print(f"\nQuestion: {question}")
        print(f"Type: {plan.query_type}")
        print(f"Routing method: {plan.routing_method}")

        for sub_query in plan.sub_queries:
            print(f"  - {sub_query}")
