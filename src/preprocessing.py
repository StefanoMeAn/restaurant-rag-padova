"""Preprocessing utilities for the Restaurant RAG Padova dataset."""

import ast
import re
from datetime import datetime

import pandas as pd
from bs4 import BeautifulSoup
from geopy.distance import distance


RELEVANT_TYPES = [
    "restaurant",
    "meal delivery",
    "bar",
    "cafe",
    "bakery",
]


PRICE_LEVELS = {
    0.0: "free",
    1.0: "inexpensive ($)",
    2.0: "moderate ($$)",
    3.0: "expensive ($$$)",
    4.0: "very expensive ($$$$)",
}


# ---------------------------------------------------------------------
# Helper functions
# ---------------------------------------------------------------------

def safe_literal_eval(value):
    """Convert a string representation of a Python object when possible."""

    if pd.isna(value):
        return None

    if not isinstance(value, str):
        return value

    try:
        return ast.literal_eval(value)
    except (ValueError, SyntaxError):
        return None


def clean_unicode(text):
    """Remove uncommon whitespace characters."""

    if not isinstance(text, str):
        return text

    return re.sub(r"[\u2009\u202f]", " ", text)


def parse_opening_hours(value):
    """Convert Google Places opening hours into a weekday dictionary."""

    data = safe_literal_eval(value)

    if not isinstance(data, dict):
        return None

    weekday_text = data.get("weekday_text")

    if not weekday_text:
        return None

    schedule = {}

    for entry in weekday_text:
        entry = clean_unicode(entry)

        if ":" in entry:
            day, hours = entry.split(":", 1)
            schedule[day] = hours.strip()

    return schedule or None


def extract_coordinates(value):
    """Extract latitude and longitude from Google Places geometry."""

    geometry = safe_literal_eval(value)

    if not isinstance(geometry, dict):
        return None

    location = geometry.get("location", {})

    lat = location.get("lat")
    lon = location.get("lng")

    if lat is None or lon is None:
        return None

    return [lat, lon]


def extract_restaurant_types(value):
    """Keep restaurant-related Google Places categories."""

    types = safe_literal_eval(value)

    if not isinstance(types, list):
        return []

    cleaned = [
        item.replace("_", " ")
        for item in types
    ]

    return [
        item
        for item in cleaned
        if item in RELEVANT_TYPES
    ]


# ---------------------------------------------------------------------
# Restaurant preprocessing
# ---------------------------------------------------------------------

def preprocess_restaurants(restaurants):
    """Clean restaurant metadata."""

    df = restaurants.copy()

    df["types"] = df["types"].apply(extract_restaurant_types)

    df["main_type"] = df["types"].apply(
        lambda x: x[0] if x else "restaurant"
    )

    df["opening_hours"] = df["opening_hours"].apply(
        parse_opening_hours
    )

    df["formatted_address"] = (
        df["formatted_address"]
        .fillna("")
        .str.replace(
            r",?\s*Padova PD, Italy",
            "",
            regex=True,
        )
    )

    df["price_level"] = df["price_level"].map(
        PRICE_LEVELS
    )

    df["geometry"] = df["geometry"].apply(
        extract_coordinates
    )

    return df


# ---------------------------------------------------------------------
# Review preprocessing
# ---------------------------------------------------------------------

def clean_review_text(text):
    """Remove HTML and normalize whitespace."""

    if pd.isna(text):
        return "User did not leave a written review."

    text = BeautifulSoup(
        str(text),
        "html.parser",
    ).get_text(" ")

    return re.sub(r"\s+", " ", text).strip()


def preprocess_reviews(reviews):
    """Clean Google Maps reviews."""

    df = reviews.copy()

    columns = [
        "place_id",
        "rating",
        "text",
        "time",
    ]

    df = df[
        [column for column in columns if column in df.columns]
    ].copy()

    df["text"] = df["text"].apply(
        clean_review_text
    )

    if "time" in df.columns:
        df["time"] = pd.to_datetime(
            df["time"],
            unit="s",
            errors="coerce",
        )

    return df


# ---------------------------------------------------------------------
# Landmark processing
# ---------------------------------------------------------------------

def find_nearest_landmarks(
    restaurant_coordinates,
    landmarks,
    max_distance=500,
    max_landmarks=3,
):
    """Find the closest landmarks to a restaurant."""

    if restaurant_coordinates is None:
        return []

    nearby = []

    for _, landmark in landmarks.iterrows():

        landmark_coordinates = landmark["geometry"]

        if landmark_coordinates is None:
            continue

        dist = distance(
            restaurant_coordinates,
            landmark_coordinates,
        ).meters

        if dist <= max_distance:
            nearby.append(
                {
                    "name": landmark["name"],
                    "distance_m": round(dist),
                }
            )

    # Closest landmarks first.
    nearby.sort(
        key=lambda item: item["distance_m"]
    )

    return nearby[:max_landmarks]


def add_nearest_landmarks(
    restaurants,
    landmarks,
    max_distance=500,
    max_landmarks=3,
):
    """Attach nearby Padova landmarks to each restaurant."""

    df = restaurants.copy()

    df["nearest_attractions"] = df["geometry"].apply(
        lambda coordinates: find_nearest_landmarks(
            coordinates,
            landmarks,
            max_distance=max_distance,
            max_landmarks=max_landmarks,
        )
    )

    return df


# ---------------------------------------------------------------------
# Main preprocessing pipeline
# ---------------------------------------------------------------------

def preprocess_dataset(restaurants, reviews):
    """Run the main preprocessing pipeline."""

    restaurants_clean = preprocess_restaurants(
        restaurants
    )

    reviews_clean = preprocess_reviews(
        reviews
    )

    return restaurants_clean, reviews_clean
