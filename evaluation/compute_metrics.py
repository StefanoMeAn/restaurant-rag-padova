"""Compare baseline and query-aware retrieval strategies."""

from src.rag_pipeline import (
    rank_restaurants,
    retrieve_documents,
)
from src.vector_store import load_vector_store


QUESTIONS = [
    {
        "id": "cheap",
        "question": "Recommend a cheap restaurant.",
        "field": "price_level",
        "expected": "inexpensive ($)",
    },
    {
        "id": "wine",
        "question": (
            "I want a restaurant with good reviews "
            "that serves wine."
        ),
        "field": "serves_wine",
        "expected": True,
    },
    {
        "id": "delivery",
        "question": (
            "Recommend a restaurant that offers delivery."
        ),
        "field": "delivery",
        "expected": True,
    },
]


def baseline_rank_restaurants(documents):
    """Reproduce the original rating-based ranking."""

    def ranking_key(document):

        rating = (
            document.metadata.get("rating")
            or 0
        )

        rating_count = (
            document.metadata.get(
                "user_ratings_total"
            )
            or 0
        )

        return (
            rating,
            rating_count,
        )

    return sorted(
        documents,
        key=ranking_key,
        reverse=True,
    )


def constraint_satisfaction(
    documents,
    field,
    expected,
):
    """Return fraction of results satisfying the constraint."""

    if not documents:
        return 0.0

    matches = sum(
        document.metadata.get(field)
        == expected
        for document in documents
    )

    return matches / len(documents)


def top1_satisfaction(
    documents,
    field,
    expected,
):
    """Return whether the first result satisfies the constraint."""

    if not documents:
        return 0

    return int(
        documents[0].metadata.get(field)
        == expected
    )


def evaluate():
    """Compare baseline and query-aware ranking."""

    print("Loading vector database...\n")

    vector_store = load_vector_store()

    rows = []

    for item in QUESTIONS:

        question = item["question"]
        field = item["field"]
        expected = item["expected"]

        # Retrieve the same semantic candidates once.
        documents = retrieve_documents(
            vector_store,
            question,
            k=4,
        )

        # Baseline:
        # rating + rating count only.
        baseline_documents = (
            baseline_rank_restaurants(
                documents.copy()
            )
        )

        # Improved:
        # query constraint + rating + rating count.
        query_aware_documents = (
            rank_restaurants(
                documents.copy(),
                question,
            )
        )

        baseline_at_4 = (
            constraint_satisfaction(
                baseline_documents,
                field,
                expected,
            )
        )

        query_aware_at_4 = (
            constraint_satisfaction(
                query_aware_documents,
                field,
                expected,
            )
        )

        baseline_at_1 = (
            top1_satisfaction(
                baseline_documents,
                field,
                expected,
            )
        )

        query_aware_at_1 = (
            top1_satisfaction(
                query_aware_documents,
                field,
                expected,
            )
        )

        rows.append(
            {
                "id": item["id"],
                "baseline_at_4":
                    baseline_at_4,
                "query_aware_at_4":
                    query_aware_at_4,
                "baseline_at_1":
                    baseline_at_1,
                "query_aware_at_1":
                    query_aware_at_1,
            }
        )

    print()
    print(
        "Restaurant RAG Retrieval Evaluation"
    )

    print("=" * 72)

    print(
        f"{'Query':<12}"
        f"{'Baseline@4':>14}"
        f"{'QueryAware@4':>16}"
        f"{'Baseline@1':>14}"
        f"{'QueryAware@1':>16}"
    )

    print("-" * 72)

    for row in rows:

        print(
            f"{row['id']:<12}"
            f"{row['baseline_at_4'] * 100:>13.1f}%"
            f"{row['query_aware_at_4'] * 100:>15.1f}%"
            f"{row['baseline_at_1'] * 100:>13.0f}%"
            f"{row['query_aware_at_1'] * 100:>15.0f}%"
        )

    print("-" * 72)

    baseline_average = sum(
        row["baseline_at_4"]
        for row in rows
    ) / len(rows)

    query_aware_average = sum(
        row["query_aware_at_4"]
        for row in rows
    ) / len(rows)

    baseline_top1_average = sum(
        row["baseline_at_1"]
        for row in rows
    ) / len(rows)

    query_aware_top1_average = sum(
        row["query_aware_at_1"]
        for row in rows
    ) / len(rows)

    print(
        f"{'Average':<12}"
        f"{baseline_average * 100:>13.1f}%"
        f"{query_aware_average * 100:>15.1f}%"
        f"{baseline_top1_average * 100:>13.1f}%"
        f"{query_aware_top1_average * 100:>15.1f}%"
    )

    print()

    print("Top results")
    print("=" * 72)

    for item in QUESTIONS:

        question = item["question"]

        documents = retrieve_documents(
            vector_store,
            question,
            k=4,
        )

        baseline_documents = (
            baseline_rank_restaurants(
                documents.copy()
            )
        )

        query_aware_documents = (
            rank_restaurants(
                documents.copy(),
                question,
            )
        )

        print(
            f"\n{item['id'].upper()}"
        )

        print(
            "Baseline:    "
            f"{baseline_documents[0].metadata.get('name')}"
        )

        print(
            "Query-aware: "
            f"{query_aware_documents[0].metadata.get('name')}"
        )


if __name__ == "__main__":
    evaluate()
