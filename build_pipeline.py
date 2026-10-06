"""Build the Restaurant RAG Padova data and retrieval pipeline."""

import pandas as pd

from src.preprocessing import preprocess_dataset
from src.corpus import build_corpus, save_corpus
from src.vector_store import build_vector_store


RESTAURANTS_PATH = "data/raw/restaurants.csv"
REVIEWS_PATH = "data/raw/reviews.csv"


def main():

    print("Loading raw dataset...")

    restaurants = pd.read_csv(RESTAURANTS_PATH)
    reviews = pd.read_csv(REVIEWS_PATH)

    print(f"Restaurants: {len(restaurants)}")
    print(f"Reviews: {len(reviews)}")

    # --------------------------------------------------------------
    # Preprocessing
    # --------------------------------------------------------------

    print("\nPreprocessing dataset...")

    restaurants_clean, reviews_clean = preprocess_dataset(
        restaurants,
        reviews,
    )

    print(
        f"Processed restaurants: {len(restaurants_clean)}"
    )
    print(
        f"Processed reviews: {len(reviews_clean)}"
    )

    # --------------------------------------------------------------
    # Corpus
    # --------------------------------------------------------------

    print("\nBuilding restaurant corpus...")

    corpus = build_corpus(
        restaurants_clean,
        reviews_clean,
    )

    print(f"Documents created: {len(corpus)}")

    save_corpus(corpus)

    print(
        "Corpus saved to "
        "data/processed/restaurants.json"
    )

    # --------------------------------------------------------------
    # Vector database
    # --------------------------------------------------------------

    print("\nBuilding vector database...")

    build_vector_store()

    print("\nPipeline completed successfully.")


if __name__ == "__main__":
    main()
