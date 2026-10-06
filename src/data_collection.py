"""Collect restaurant data from the Google Places API.

This module reproduces the data-collection strategy used for the
Restaurant RAG Padova project. Padova is divided into overlapping
search regions, restaurants are discovered with Nearby Search, and
their detailed information and reviews are retrieved with Place Details.
"""

import os
import time
from math import cos, radians
from pathlib import Path

import pandas as pd
import requests
from dotenv import load_dotenv


# ---------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------

load_dotenv()

API_KEY = os.getenv("GOOGLE_MAPS_API_KEY")

NEARBY_SEARCH_URL = (
    "https://maps.googleapis.com/maps/api/place/nearbysearch/json"
)

PLACE_DETAILS_URL = (
    "https://maps.googleapis.com/maps/api/place/details/json"
)

PADOVA_LAT = 45.4064
PADOVA_LON = 11.8768

SEARCH_RADIUS = 500       # metres
GRID_DISPLACEMENT = 1000  # metres

PLACE_FIELDS = [
    "place_id",
    "name",
    "types",
    "formatted_address",
    "rating",
    "user_ratings_total",
    "geometry",
    "website",
    "formatted_phone_number",
    "price_level",
    "delivery",
    "dine_in",
    "reservable",
    "serves_breakfast",
    "serves_lunch",
    "serves_dinner",
    "serves_beer",
    "serves_wine",
    "opening_hours",
]

DETAIL_FIELDS = PLACE_FIELDS + ["reviews"]


# ---------------------------------------------------------------------
# Grid generation
# ---------------------------------------------------------------------

def compute_position(dx, dy, lat, lon):
    """Return latitude and longitude after a displacement in metres."""

    delta_lon = dx / (111320 * cos(radians(lat)))
    delta_lat = dy / 110540

    return (
        round(lat + delta_lat, 5),
        round(lon + delta_lon, 5),
    )


def create_search_grid(
    center_lat=PADOVA_LAT,
    center_lon=PADOVA_LON,
    displacement=GRID_DISPLACEMENT,
):
    """Create overlapping search points covering central Padova."""

    grid = []

    for i in range(-2, 3):
        for j in range(-2, 3):

            grid.append(
                compute_position(
                    displacement * j,
                    -displacement * i,
                    center_lat,
                    center_lon,
                )
            )

            grid.append(
                compute_position(
                    displacement / 2 + displacement * j,
                    -displacement / 2 - displacement * i,
                    center_lat,
                    center_lon,
                )
            )

    return grid


# ---------------------------------------------------------------------
# Google Places API
# ---------------------------------------------------------------------

def discover_restaurants(grid, radius=SEARCH_RADIUS):
    """Discover restaurant place IDs across the search grid."""

    place_ids = set()

    for lat, lon in grid:

        parameters = {
            "location": f"{lat},{lon}",
            "radius": radius,
            "type": "restaurant",
            "key": API_KEY,
        }

        while True:

            response = requests.get(
                NEARBY_SEARCH_URL,
                params=parameters,
                timeout=30,
            )

            response.raise_for_status()
            data = response.json()

            for place in data.get("results", []):
                place_ids.add(place["place_id"])

            next_page_token = data.get("next_page_token")

            if not next_page_token:
                break

            # Google requires a short delay before the next-page
            # token becomes active.
            time.sleep(2)

            parameters = {
                "pagetoken": next_page_token,
                "key": API_KEY,
            }

    return sorted(place_ids)


def get_place_details(place_id):
    """Retrieve restaurant metadata and reviews."""

    parameters = {
        "place_id": place_id,
        "fields": ",".join(DETAIL_FIELDS),
        "key": API_KEY,
    }

    response = requests.get(
        PLACE_DETAILS_URL,
        params=parameters,
        timeout=30,
    )

    response.raise_for_status()

    return response.json().get("result")


# ---------------------------------------------------------------------
# Dataset creation
# ---------------------------------------------------------------------

def collect_dataset(place_ids):
    """Collect restaurant metadata and reviews."""

    restaurants = []
    reviews = []

    for place_id in place_ids:

        result = get_place_details(place_id)

        if not result:
            continue

        restaurant = {
            field: result.get(field)
            for field in PLACE_FIELDS
        }

        restaurants.append(restaurant)

        for review in result.get("reviews", []):
            review_data = review.copy()
            review_data["place_id"] = place_id
            reviews.append(review_data)

    restaurants_df = pd.DataFrame(restaurants)
    reviews_df = pd.DataFrame(reviews)

    # Keep unique Google Places.
    restaurants_df = restaurants_df.drop_duplicates(
        subset="place_id"
    )

    # Keep restaurants located in Padova.
    restaurants_df = restaurants_df[
        restaurants_df["formatted_address"]
        .fillna("")
        .str.contains("Padova", case=False)
    ].reset_index(drop=True)

    valid_place_ids = set(restaurants_df["place_id"])

    if not reviews_df.empty:
        reviews_df = reviews_df[
            reviews_df["place_id"].isin(valid_place_ids)
        ].reset_index(drop=True)

    return restaurants_df, reviews_df


# ---------------------------------------------------------------------
# Save data
# ---------------------------------------------------------------------

def save_dataset(restaurants, reviews, output_dir="data/raw"):
    """Save collected data as CSV files."""

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    restaurants.to_csv(
        output_path / "restaurants.csv",
        index=False,
    )

    reviews.to_csv(
        output_path / "reviews.csv",
        index=False,
    )


# ---------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------

def main():

    if not API_KEY:
        raise RuntimeError(
            "GOOGLE_MAPS_API_KEY is not defined. "
            "Create a .env file with your API key."
        )

    print("Creating Padova search grid...")
    grid = create_search_grid()

    print(f"Searching {len(grid)} regions...")
    place_ids = discover_restaurants(grid)

    print(f"Found {len(place_ids)} unique places.")

    print("Retrieving restaurant details and reviews...")
    restaurants, reviews = collect_dataset(place_ids)

    print(f"Restaurants in dataset: {len(restaurants)}")
    print(f"Reviews in dataset: {len(reviews)}")

    save_dataset(restaurants, reviews)

    print("Dataset saved to data/raw/")


if __name__ == "__main__":
    main()
