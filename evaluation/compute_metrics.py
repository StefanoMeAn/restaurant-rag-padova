"""Compare baseline, query-aware v1, and query-aware v2 retrieval."""

from src.rag_pipeline import (
    CANDIDATE_K,
    rank_restaurants,
    retrieve_documents,
)
from src.vector_store import load_vector_store


TOP_K = 4

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
        rating = document.metadata.get("rating") or 0
        rating_count = (
            document.metadata.get("user_ratings_total")
            or 0
        )

        return rating, rating_count

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
    """Return fraction of results satisfying a constraint."""

    if not documents:
        return 0.0

    matches = sum(
        document.metadata.get(field) == expected
        for document in documents
    )

    return matches / len(documents)


def top1_satisfaction(
    documents,
    field,
    expected,
):
    """Return whether the first result satisfies a constraint."""

    if not documents:
        return 0

    return int(
        documents[0].metadata.get(field)
        == expected
    )


def evaluate_strategy(
    documents,
    field,
    expected,
):
    """Calculate @4 and @1 constraint metrics."""

    return {
        "at_4": constraint_satisfaction(
            documents,
            field,
            expected,
        ),
        "at_1": top1_satisfaction(
            documents,
            field,
            expected,
        ),
    }


def evaluate():
    """Compare the three retrieval strategies."""

    print("Loading vector database...\n")

    vector_store = load_vector_store()

    rows = []

    for item in QUESTIONS:

        question = item["question"]
        field = item["field"]
        expected = item["expected"]

        # -------------------------------------------------
        # Baseline and v1 use the same four semantic results
        # -------------------------------------------------

        top4_candidates = retrieve_documents(
            vector_store,
            question,
            k=TOP_K,
        )

        baseline_documents = (
            baseline_rank_restaurants(
                top4_candidates.copy()
            )
        )

        v1_documents = rank_restaurants(
            top4_candidates.copy(),
            question,
        )

        # -------------------------------------------------
        # v2 retrieves a larger candidate pool first
        # -------------------------------------------------

        candidate_pool = retrieve_documents(
            vector_store,
            question,
            k=CANDIDATE_K,
        )

        v2_documents = rank_restaurants(
            candidate_pool,
            question,
        )[:TOP_K]

        baseline_metrics = evaluate_strategy(
            baseline_documents,
            field,
            expected,
        )

        v1_metrics = evaluate_strategy(
            v1_documents,
            field,
            expected,
        )

        v2_metrics = evaluate_strategy(
            v2_documents,
            field,
            expected,
        )

        rows.append(
            {
                "id": item["id"],
                "baseline": baseline_metrics,
                "v1": v1_metrics,
                "v2": v2_metrics,
                "baseline_top":
                    baseline_documents[0].metadata.get(
                        "name"
                    ),
                "v1_top":
                    v1_documents[0].metadata.get(
                        "name"
                    ),
                "v2_top":
                    v2_documents[0].metadata.get(
                        "name"
                    ),
            }
        )

    print(
        "Restaurant RAG Retrieval Evaluation"
    )
    print("=" * 75)

    print(
        f"{'Query':<12}"
        f"{'Baseline@4':>13}"
        f"{'V1@4':>10}"
        f"{'V2@4':>10}"
        f"{'Baseline@1':>14}"
        f"{'V1@1':>10}"
        f"{'V2@1':>10}"
    )

    print("-" * 75)

    for row in rows:

        print(
            f"{row['id']:<12}"
            f"{row['baseline']['at_4'] * 100:>12.1f}%"
            f"{row['v1']['at_4'] * 100:>9.1f}%"
            f"{row['v2']['at_4'] * 100:>9.1f}%"
            f"{row['baseline']['at_1'] * 100:>13.0f}%"
            f"{row['v1']['at_1'] * 100:>9.0f}%"
            f"{row['v2']['at_1'] * 100:>9.0f}%"
        )

    print("-" * 75)

    for metric in ["at_4", "at_1"]:

        baseline_average = sum(
            row["baseline"][metric]
            for row in rows
        ) / len(rows)

        v1_average = sum(
            row["v1"][metric]
            for row in rows
        ) / len(rows)

        v2_average = sum(
            row["v2"][metric]
            for row in rows
        ) / len(rows)

        label = (
            "Average@4"
            if metric == "at_4"
            else "Average@1"
        )

        print(
            f"{label:<12}"
            f"{baseline_average * 100:>12.1f}%"
            f"{v1_average * 100:>9.1f}%"
            f"{v2_average * 100:>9.1f}%"
        )

    print()
    print("Top-ranked restaurants")
    print("=" * 75)

    for row in rows:

        print(f"\n{row['id'].upper()}")
        print(
            f"Baseline: {row['baseline_top']}"
        )
        print(
            f"V1:       {row['v1_top']}"
        )
        print(
            f"V2:       {row['v2_top']}"
        )


if __name__ == "__main__":
    evaluate()
