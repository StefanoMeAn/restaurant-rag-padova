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
OUTPUT_PATH = RESULTS_DIR / "query_aware_retrieval_v1.json"


def load_questions():
    """Load evaluation questions."""

    with open(
        QUESTIONS_PATH,
        "r",
        encoding="utf-8",
    ) as file:
        return json.load(file)


def evaluate():
    """Run retrieval evaluation and save the results."""

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

        # Semantic retrieval
        documents = retrieve_documents(
            vector_store,
            question,
            k=4,
        )

        # Query-aware ranking
        documents = rank_restaurants(
            documents,
            question,
        )

        retrieved = []

        print("=" * 80)
        print(f"[{category.upper()}] {question}")
        print()

        for rank, document in enumerate(
            documents,
            start=1,
        ):

            metadata = document.metadata

            name = metadata.get(
                "name",
                "Unknown",
            )

            place_id = metadata.get(
                "place_id"
            )

            rating = metadata.get(
                "rating"
            )

            rating_count = metadata.get(
                "user_ratings_total"
            )

            price = metadata.get(
                "price_level"
            )

            location = metadata.get(
                "location"
            )

            delivery = metadata.get(
                "delivery"
            )

            dine_in = metadata.get(
                "dine_in"
            )

            reservable = metadata.get(
                "reservable"
            )

            serves_wine = metadata.get(
                "serves_wine"
            )

            serves_beer = metadata.get(
                "serves_beer"
            )

            chunk_id = metadata.get(
                "chunk_id"
            )

            restaurant = {
                "rank": rank,
                "name": name,
                "place_id": place_id,
                "rating": rating,
                "user_ratings_total": rating_count,
                "price_level": price,
                "location": location,
                "delivery": delivery,
                "dine_in": dine_in,
                "reservable": reservable,
                "serves_wine": serves_wine,
                "serves_beer": serves_beer,
                "chunk_id": chunk_id,
            }

            retrieved.append(
                restaurant
            )

            print(f"{rank}. {name}")

            print(
                f"   Rating: {rating}/5 "
                f"({rating_count} ratings)"
            )

            print(
                f"   Price: {price}"
            )

            print(
                f"   Delivery: {delivery}"
            )

            print(
                f"   Dine-in: {dine_in}"
            )

            print(
                f"   Reservable: {reservable}"
            )

            print(
                f"   Serves wine: {serves_wine}"
            )

            print(
                f"   Serves beer: {serves_beer}"
            )

            print(
                f"   Location: {location}"
            )

            print()

        results.append(
            {
                "id": item["id"],
                "category": category,
                "question": question,
                "retrieved_restaurants": retrieved,
            }
        )

    # Create results directory if necessary
    RESULTS_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    # Save evaluation results
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
        "Query-aware retrieval results saved to:"
    )

    print(
        OUTPUT_PATH
    )


if __name__ == "__main__":
    evaluate()
