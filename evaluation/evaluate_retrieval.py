"""Evaluate retrieval quality for the Restaurant RAG Padova system."""

import json
from pathlib import Path

from src.rag_pipeline import (
    rank_restaurants,
    retrieve_documents,
)
from src.vector_store import load_vector_store


QUESTIONS_PATH = Path("evaluation/questions.json")


def load_questions():
    """Load evaluation questions."""

    with open(
        QUESTIONS_PATH,
        "r",
        encoding="utf-8",
    ) as file:
        return json.load(file)


def evaluate():
    """Run all evaluation questions through the retriever."""

    print("Loading vector database...\n")

    vector_store = load_vector_store()

    questions = load_questions()

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
            documents
        )

        print("=" * 80)
        print(
            f"[{category.upper()}] {question}"
        )
        print()

        for index, document in enumerate(
            documents,
            start=1,
        ):
            metadata = document.metadata

            name = metadata.get(
                "name",
                "Unknown",
            )

            rating = metadata.get(
                "rating",
                "N/A",
            )

            rating_count = metadata.get(
                "user_ratings_total",
                "N/A",
            )

            price = metadata.get(
                "price_level",
                "N/A",
            )

            print(
                f"{index}. {name}"
            )

            print(
                f"   Rating: {rating}/5 "
                f"({rating_count} ratings)"
            )

            print(
                f"   Price: {price}"
            )

        print()


if __name__ == "__main__":
    evaluate()
