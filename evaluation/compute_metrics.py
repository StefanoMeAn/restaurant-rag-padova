"""Compute retrieval metrics for Restaurant RAG Padova."""

import json
from pathlib import Path


RESULTS_DIR = Path("evaluation/results")

BASELINE_PATH = (
    RESULTS_DIR / "baseline_retrieval_v1.json"
)

QUERY_AWARE_PATH = (
    RESULTS_DIR / "query_aware_retrieval_v1.json"
)


CONSTRAINTS = {
    "cheap": {
        "field": "price_level",
        "expected": "inexpensive ($)",
    },
    "wine": {
        "field": "serves_wine",
        "expected": True,
    },
    "delivery": {
        "field": "delivery",
        "expected": True,
    },
}


def load_results(path):
    """Load evaluation results from JSON."""

    with open(
        path,
        "r",
        encoding="utf-8",
    ) as file:
        return json.load(file)


def find_question(results, question_id):
    """Find evaluation results for a specific question."""

    for item in results:
        if item["id"] == question_id:
            return item

    raise ValueError(
        f"Question '{question_id}' not found."
    )


def constraint_satisfaction(
    item,
    field,
    expected,
):
    """Calculate the fraction of top-k results satisfying a constraint."""

    restaurants = item[
        "retrieved_restaurants"
    ]

    if not restaurants:
        return 0.0

    matches = sum(
        restaurant.get(field) == expected
        for restaurant in restaurants
    )

    return matches / len(restaurants)


def top1_satisfaction(
    item,
    field,
    expected,
):
    """Check whether the top-ranked restaurant satisfies the constraint."""

    restaurants = item[
        "retrieved_restaurants"
    ]

    if not restaurants:
        return 0

    return int(
        restaurants[0].get(field)
        == expected
    )


def evaluate_system(
    results,
):
    """Compute constraint metrics for one retrieval system."""

    metrics = {}

    for question_id, constraint in CONSTRAINTS.items():

        item = find_question(
            results,
            question_id,
        )

        field = constraint["field"]
        expected = constraint["expected"]

        satisfaction_at_4 = (
            constraint_satisfaction(
                item,
                field,
                expected,
            )
        )

        top1 = top1_satisfaction(
            item,
            field,
            expected,
        )

        metrics[question_id] = {
            "constraint_satisfaction_at_4":
                satisfaction_at_4,
            "top1_constraint_satisfaction":
                top1,
        }

    return metrics


def print_comparison(
    baseline_metrics,
    query_aware_metrics,
):
    """Print baseline vs query-aware retrieval metrics."""

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

    for question_id in CONSTRAINTS:

        baseline = baseline_metrics[
            question_id
        ]

        improved = query_aware_metrics[
            question_id
        ]

        baseline_at_4 = (
            baseline[
                "constraint_satisfaction_at_4"
            ]
            * 100
        )

        improved_at_4 = (
            improved[
                "constraint_satisfaction_at_4"
            ]
            * 100
        )

        baseline_at_1 = (
            baseline[
                "top1_constraint_satisfaction"
            ]
            * 100
        )

        improved_at_1 = (
            improved[
                "top1_constraint_satisfaction"
            ]
            * 100
        )

        print(
            f"{question_id:<12}"
            f"{baseline_at_4:>13.1f}%"
            f"{improved_at_4:>15.1f}%"
            f"{baseline_at_1:>13.0f}%"
            f"{improved_at_1:>15.0f}%"
        )

    print("-" * 72)

    baseline_average = sum(
        value[
            "constraint_satisfaction_at_4"
        ]
        for value in baseline_metrics.values()
    ) / len(baseline_metrics)

    query_aware_average = sum(
        value[
            "constraint_satisfaction_at_4"
        ]
        for value in query_aware_metrics.values()
    ) / len(query_aware_metrics)

    print(
        f"{'Average':<12}"
        f"{baseline_average * 100:>13.1f}%"
        f"{query_aware_average * 100:>15.1f}%"
    )


def main():
    """Compare baseline and query-aware retrieval."""

    baseline_results = load_results(
        BASELINE_PATH
    )

    query_aware_results = load_results(
        QUERY_AWARE_PATH
    )

    baseline_metrics = evaluate_system(
        baseline_results
    )

    query_aware_metrics = evaluate_system(
        query_aware_results
    )

    print_comparison(
        baseline_metrics,
        query_aware_metrics,
    )


if __name__ == "__main__":
    main()
