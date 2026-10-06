"""Build textual documents for the Restaurant RAG Padova system."""

import json
from pathlib import Path

import pandas as pd


# ---------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------

def is_true(value):
    """Return True only for boolean-like true values."""

    if pd.isna(value):
        return False

    if isinstance(value, bool):
        return value

    if isinstance(value, str):
        return value.strip().lower() == "true"

    return bool(value)


# ---------------------------------------------------------------------
# Restaurant information
# ---------------------------------------------------------------------

def format_restaurant_description(row):
    """Create the main textual description of a restaurant."""

    name = row.get("name", "Unknown restaurant")
    restaurant_type = row.get("main_type", "restaurant")
    address = row.get("formatted_address", "Unknown address")

    text = (
        f"{name} is a {restaurant_type} located at {address}.\n"
    )

    rating = row.get("rating")
    rating_count = row.get("user_ratings_total")

    if pd.notna(rating):
        text += f"It has an average rating of {rating}/5"

        if pd.notna(rating_count):
            text += f" based on {int(rating_count)} user ratings"

        text += ".\n"

    price = row.get("price_level")

    if pd.notna(price):
        text += f"Price level: {price}.\n"

    return text


# ---------------------------------------------------------------------
# Services
# ---------------------------------------------------------------------

def format_services(row):
    """Describe available restaurant services."""

    services = []

    service_columns = {
        "delivery": "delivery",
        "dine_in": "dine-in",
        "reservable": "reservations",
    }

    for column, label in service_columns.items():
        if is_true(row.get(column)):
            services.append(label)

    if not services:
        return ""

    return "Services: " + ", ".join(services) + ".\n"


def format_serving_options(row):
    """Describe meals and drinks served by the restaurant."""

    options = []

    serving_columns = {
        "serves_breakfast": "breakfast",
        "serves_lunch": "lunch",
        "serves_dinner": "dinner",
        "serves_beer": "beer",
        "serves_wine": "wine",
    }

    for column, label in serving_columns.items():
        if is_true(row.get(column)):
            options.append(label)

    if not options:
        return ""

    return "Serves: " + ", ".join(options) + ".\n"


# ---------------------------------------------------------------------
# Location
# ---------------------------------------------------------------------

def format_landmarks(row):
    """Describe nearby Padova landmarks."""

    landmarks = row.get("nearest_attractions")

    if landmarks is None:
        return ""

    if isinstance(landmarks, float) and pd.isna(landmarks):
        return ""

    if not landmarks:
        return ""

    entries = []

    for landmark in landmarks:
        entries.append(
            f"{landmark['name']} ({landmark['distance_m']} m)"
        )

    return "Nearby landmarks: " + ", ".join(entries) + ".\n"


# ---------------------------------------------------------------------
# Opening hours
# ---------------------------------------------------------------------

def format_opening_hours(row):
    """Format weekly opening hours."""

    schedule = row.get("opening_hours")

    if not isinstance(schedule, dict) or not schedule:
        return ""

    lines = ["Opening hours:"]

    days = [
        "Monday",
        "Tuesday",
        "Wednesday",
        "Thursday",
        "Friday",
        "Saturday",
        "Sunday",
    ]

    for day in days:
        hours = schedule.get(day, "Closed")
        lines.append(f"- {day}: {hours}")

    return "\n".join(lines) + "\n"


# ---------------------------------------------------------------------
# Reviews
# ---------------------------------------------------------------------

def format_reviews(place_id, reviews):
    """Format all available reviews for a restaurant."""

    restaurant_reviews = reviews[
        reviews["place_id"] == place_id
    ]

    if restaurant_reviews.empty:
        return "Reviews: No written reviews available.\n"

    lines = ["Reviews:"]

    for _, review in restaurant_reviews.iterrows():

        rating = review.get("rating")
        text = review.get("text")
        date = review.get("time")

        if pd.notna(date):
            date = str(date).split(" ")[0]
        else:
            date = "Unknown date"

        if pd.isna(rating):
            rating_text = "Unknown rating"
        else:
            rating_text = f"{rating}/5"

        lines.append(
            f"- {rating_text}, {date}: {text}"
        )

    return "\n".join(lines) + "\n"


# ---------------------------------------------------------------------
# Complete restaurant document
# ---------------------------------------------------------------------

def build_restaurant_document(row, reviews):
    """Build the complete RAG document for one restaurant."""

    sections = [
        format_restaurant_description(row),
        format_services(row),
        format_serving_options(row),
        format_landmarks(row),
        format_opening_hours(row),
        format_reviews(
            row["place_id"],
            reviews,
        ),
    ]

    return "\n".join(
        section.strip()
        for section in sections
        if section and section.strip()
    )


# ---------------------------------------------------------------------
# Corpus
# ---------------------------------------------------------------------

def build_corpus(restaurants, reviews):
    """Build one document and metadata dictionary per restaurant."""

    documents = []

    for _, row in restaurants.iterrows():

        document = {
            "text": build_restaurant_document(
                row,
                reviews,
            ),
            "metadata": {
                "place_id": row.get("place_id"),
                "name": row.get("name"),
                "type": row.get("main_type"),
                "location": row.get("formatted_address"),
                "rating": row.get("rating"),
                "user_ratings_total": row.get(
                    "user_ratings_total"
                ),
                "price_level": row.get("price_level"),
                "serves_wine": is_true(
                    row.get("serves_wine")
                ),
            },
        }

        documents.append(document)

    return documents


# ---------------------------------------------------------------------
# Save corpus
# ---------------------------------------------------------------------

def save_corpus(
    documents,
    output_file="data/processed/restaurants.json",
):
    """Save the generated restaurant corpus as JSON."""

    output_path = Path(output_file)

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with open(
        output_path,
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            documents,
            file,
            ensure_ascii=False,
            indent=2,
            default=str,
        )
