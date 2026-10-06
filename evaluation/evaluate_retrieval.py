"""Evaluate retrieval quality for the Restaurant RAG Padova system."""

import json
from pathlib import Path

from src.rag_pipeline import (
    rank_restaurants,
    retrieve_documents,
)
from src.vector_store import load_vector_store


QUESTIONS_PATH = Path("evaluation/questions.json")
RESULTS_DIR = Path("evaluation/results")
OUTPUT_PATH = RESULTS_DIR / "baseline_retrieval.json"


def load_questions():
    """Load evaluation questions."""

    with open(
        QUESTIONS_PATH,
        "r",
        encoding="utf-8",
    ) as file:
        return json.load(file)


def evaluate():
    """Run evaluation questions and save retrieval results."""

    print("Loading vector database...\n")

    vector_store = load_vector_store()
    questions = load_questions()

    results = []

    print(
        f"Running retrieval evaluation "
        f"on {len(questions)} questions...\n"
    )

    for item in questions:

        question = item["question"]
        category = item["category"]

        documents = retrieve_documents(
            vector_store,
            question,
            k=4,
        )

        documents = rank_restaurants(
            documents,
            question
        )

        retrieved = []

        print("=" * 80)
        print(f"[{category.upper()}] {question}\n")

        for rank, document in enumerate(
            documents,
            start=1,
        ):

            metadata = document.metadata

            restaurant = {
                "rank": rank,
                "name": metadata.get("name"),
                "place_id": metadata.get("place_id"),
                "rating": metadata.get("rating"),
                "user_ratings_total": metadata.get(
                    "user_ratings_total"
                ),
                "price_level": metadata.get("price_level"),
                "location": metadata.get("location"),
                "chunk_id": metadata.get("chunk_id"),
            }

            retrieved.append(restaurant)

            print(f"{rank}. {restaurant['name']}")
            print(
                f"   Rating: {restaurant['rating']}/5 "
                f"({restaurant['user_ratings_total']} ratings)"
            )
            print(
                f"   Price: {restaurant['price_level']}"
            )

        results.append(
            {
                "id": item["id"],
                "category": category,
                "question": question,
                "retrieved_restaurants": retrieved,
            }
        )

        print()

    RESULTS_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    with open(
        OUTPUT_PATH,
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            results,
            file,
            ensure_ascii=False,
            indent=2,
        )

    print("=" * 80)
    print(
        f"Baseline results saved to: {OUTPUT_PATH}"
    )


if __name__ == "__main__":
    evaluate()
