import os
import re
from pathlib import Path

from dotenv import load_dotenv
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import (
    RunnableLambda,
    RunnablePassthrough,
)
from langchain_openrouter import ChatOpenRouter

from backend.app.rag.query_router import route_query
from backend.app.rag.reranker import rerank_documents
from backend.app.retrieval.query_index import load_vector_store

# Load the .env file from the project root.
PROJECT_ROOT = Path(__file__).resolve().parents[3]
load_dotenv(PROJECT_ROOT / ".env")


def create_llm() -> ChatOpenRouter:
    """Create the OpenRouter chat model with sufficient token budget for reasoning models."""

    api_key = os.getenv("OPENROUTER_API_KEY")
    if not api_key:
        raise RuntimeError(
            "OPENROUTER_API_KEY is missing. Add it to the project-root .env file."
        )

    model_name = os.getenv(
        "OPENROUTER_MODEL",
        "openrouter/free",
    ).strip()

    if not model_name:
        raise RuntimeError(
            "OPENROUTER_MODEL is empty. Set it in the project-root .env file."
        )

    max_tokens = int(os.getenv("OPENROUTER_MAX_TOKENS", "4096"))

    return ChatOpenRouter(
        model=model_name,
        api_key=api_key,
        temperature=0,
        max_tokens=max_tokens,
        max_retries=2,
    )


def parse_llm_response(message) -> str:
    """Extract and clean response content from the LLM, handling reasoning models and empty text."""
    if hasattr(message, "content"):
        content = message.content
        if isinstance(content, str) and content.strip():
            cleaned = re.sub(r"<think>.*?</think>", "", content, flags=re.DOTALL).strip()
            if cleaned:
                return cleaned
            return content.strip()
        elif isinstance(content, list):
            texts = []
            for item in content:
                if isinstance(item, dict) and item.get("type") == "text":
                    texts.append(item.get("text", ""))
                elif isinstance(item, str):
                    texts.append(item)
            joined = "\n".join(texts).strip()
            if joined:
                return joined

    # If content was empty due to token exhaustion on reasoning models, salvage reasoning_content
    if hasattr(message, "additional_kwargs"):
        reasoning = message.additional_kwargs.get("reasoning_content", "")
        if isinstance(reasoning, str) and reasoning.strip():
            return reasoning.strip()

    return "No explicit answer could be generated. Please review the retrieved source citations below."


def format_document(document) -> str:
    """Include document metadata in the retrieved context."""

    metadata = document.metadata
    source = Path(str(metadata.get("source", "Unknown"))).name
    page = int(metadata.get("page", 0)) + 1
    chunk_id = metadata.get("chunk_id", "Unknown")

    return (
        f"[Source: {source}, Page: {page}, "
        f"Chunk ID: {chunk_id}]\n"
        f"{document.page_content}"
    )


def build_rag_chain(llm, k: int = 4, summary_k: int = 8, vector_store=None):
    """Build an LCEL RAG chain with routing and cross-encoder reranking."""

    if k < 1 or summary_k < 1:
        raise ValueError("Retrieval counts must be at least 1.")

    if vector_store is None:
        vector_store = load_vector_store()

    prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                """
You are a careful HR Policy Assistant.

Answer the original question using only the retrieved HR
policy context.

Rules:
1. Do not invent policies or employee entitlements.
2. Answer every part of a multi-part question.
3. If the context lacks the answer, say you could not find it
   in the provided HR policy documents.
4. Cite source filenames and page numbers when available.
5. State limitations when the context is incomplete or ambiguous.

Question type: {query_type}

Sub-questions:
{sub_query_text}

Retrieved context:
{context}
            """,
            ),
            ("human", "{question}"),
        ]
    )

    def retrieve_context(inputs: dict) -> dict:
        question = inputs["question"].strip()
        if not question:
            raise ValueError("Question cannot be empty.")

        plan = route_query(question, llm=llm)

        search_k = summary_k if plan.query_type == "summarization" else k

        candidate_k = max(search_k * 3, 10)

        all_documents = []
        seen_keys = set()
        retrieval_trace = []

        for sub_question in plan.sub_queries:
            # 1. Retrieve candidate chunks from FAISS.
            faiss_results = vector_store.similarity_search_with_score(
                sub_question,
                k=candidate_k,
            )
            candidates = [document for document, _ in faiss_results]

            # 2. Rerank those candidates using the cross-encoder.
            reranked_results = rerank_documents(
                sub_question,
                candidates,
            )

            # 3. Only the top reranked chunks are used as context.
            selected_results = reranked_results[:search_k]
            selected_documents = [document for document, _ in selected_results]

            # Keep both rankings for evidence and debugging.
            retrieval_trace.append(
                {
                    "sub_query": sub_question,
                    "faiss_results": faiss_results,
                    "reranked_results": reranked_results,
                    "selected_documents": selected_documents,
                }
            )

            # 4. Merge selected chunks, removing duplicates.
            for document in selected_documents:
                metadata = document.metadata
                key = (
                    str(metadata.get("source", "")),
                    metadata.get("page"),
                    metadata.get("chunk_id"),
                    document.page_content,
                )

                if key not in seen_keys:
                    seen_keys.add(key)
                    all_documents.append(document)

        context = "\n\n---\n\n".join(
            format_document(document) for document in all_documents
        )

        return {
            "question": question,
            "query_type": plan.query_type,
            "routing_method": plan.routing_method,
            "sub_queries": plan.sub_queries,
            "sub_query_text": "\n".join(f"- {item}" for item in plan.sub_queries),
            "context": context,
            "source_documents": all_documents,
            "retrieval_trace": retrieval_trace,
        }

    def format_output(result: dict) -> dict:
        sources = []

        for document in result["source_documents"]:
            metadata = document.metadata
            sources.append(
                {
                    "source": Path(str(metadata.get("source", "Unknown"))).name,
                    "page": int(metadata.get("page", 0)) + 1,
                    "chunk_id": metadata.get("chunk_id"),
                    "snippet": document.page_content[:300],
                }
            )

        trace = []

        for entry in result["retrieval_trace"]:
            before = []
            after = []

            for rank, (document, distance) in enumerate(
                entry["faiss_results"], start=1
            ):
                metadata = document.metadata
                before.append(
                    {
                        "rank": rank,
                        "source": Path(str(metadata.get("source", "Unknown"))).name,
                        "page": int(metadata.get("page", 0)) + 1,
                        "chunk_id": metadata.get("chunk_id"),
                        "score": float(distance),
                        "snippet": document.page_content[:180],
                    }
                )

            for rank, (document, score) in enumerate(
                entry["reranked_results"], start=1
            ):
                metadata = document.metadata
                after.append(
                    {
                        "rank": rank,
                        "source": Path(str(metadata.get("source", "Unknown"))).name,
                        "page": int(metadata.get("page", 0)) + 1,
                        "chunk_id": metadata.get("chunk_id"),
                        "score": float(score),
                        "snippet": document.page_content[:180],
                    }
                )

            before_top = (
                (before[0]["source"], before[0]["page"], before[0]["chunk_id"])
                if before
                else None
            )
            after_top = (
                (after[0]["source"], after[0]["page"], after[0]["chunk_id"])
                if after
                else None
            )

            trace.append(
                {
                    "sub_query": entry["sub_query"],
                    "before_reranking": before,
                    "after_reranking": after,
                    "top_result_changed": before_top != after_top,
                }
            )

        raw_answer = result.get("answer", "")
        clean_answer = raw_answer.strip() if isinstance(raw_answer, str) else str(raw_answer)
        if not clean_answer:
            clean_answer = (
                "No explicit answer could be generated. "
                "Please review the retrieved source citations below."
            )

        return {
            "answer": clean_answer,
            "query_type": result["query_type"],
            "routing_method": result["routing_method"],
            "sub_queries": result["sub_queries"],
            "sources": sources,
            "retrieval_trace": trace,
            "retrieved_contexts": [
                document.page_content for document in result["source_documents"]
            ],
        }

    return (
        RunnableLambda(retrieve_context)
        | RunnablePassthrough.assign(answer=prompt | llm | RunnableLambda(parse_llm_response))
        | RunnableLambda(format_output)
    )


def main():
    question = (
        "How many casual leaves are allocated on a half-yearly "
        "basis, and when are the two half-yearly cycles?"
    )

    chain = build_rag_chain(create_llm(), k=4)
    result = chain.invoke({"question": question})

    print("\nQUESTION")
    print(question)

    print("\nQUERY TYPE")
    print(result["query_type"])

    print("\nROUTING METHOD")
    print(result["routing_method"])

    print("\nSUB-QUESTIONS")
    for sub_question in result["sub_queries"]:
        print(f"- {sub_question}")

    print("\nANSWER")
    print(result["answer"])

    print("\nRETRIEVAL TRACE")

    print("\nRERANKING EVIDENCE")

    for item in result["retrieval_trace"]:
        print(f"\nSub-question: {item['sub_query']}")

        print("Before reranking — FAISS order:")
        for chunk in item["before_reranking"][:5]:
            print(
                f"  {chunk['rank']}. {chunk['source']} | "
                f"Page {chunk['page']} | Chunk {chunk['chunk_id']} | "
                f"L2 distance: {chunk['score']:.4f}"
            )

        print("After reranking — cross-encoder order:")
        for chunk in item["after_reranking"][:5]:
            print(
                f"  {chunk['rank']}. {chunk['source']} | "
                f"Page {chunk['page']} | Chunk {chunk['chunk_id']} | "
                f"Score: {chunk['score']:.4f}"
            )

        print("Top result changed:", item["top_result_changed"])
        print("Selected for context:")
        for chunk in item["after_reranking"][:4]:
            print(
                f"  - {chunk['source']} | "
                f"Page {chunk['page']} | Chunk {chunk['chunk_id']}"
            )

    print("\nFINAL SOURCES")
    for source in result["sources"]:
        print(
            f"- {source['source']} | Page {source['page']} | Chunk {source['chunk_id']}"
        )


if __name__ == "__main__":
    main()
