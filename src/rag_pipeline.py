"""RAG pipeline for restaurant recommendations in Padova."""

import torch
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    pipeline,
)

from src.vector_store import load_vector_store


READER_MODEL_NAME = "microsoft/Phi-3-mini-4k-instruct"
TOP_K = 4


def load_llm():
    """Load Phi-3 Mini for answer generation."""

    use_cuda = torch.cuda.is_available()

    tokenizer = AutoTokenizer.from_pretrained(
        READER_MODEL_NAME
    )

    if use_cuda:

        quantization_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
        )

        model = AutoModelForCausalLM.from_pretrained(
            READER_MODEL_NAME,
            quantization_config=quantization_config,
            device_map="auto",
        )

    else:

        model = AutoModelForCausalLM.from_pretrained(
            READER_MODEL_NAME,
            device_map="cpu",
            torch_dtype=torch.float32,
        )

    generator = pipeline(
        task="text-generation",
        model=model,
        tokenizer=tokenizer,
        max_new_tokens=150,
        temperature=0.3,
        top_p=0.8,
        repetition_penalty=1.1,
        return_full_text=False,
    )

    return tokenizer, generator


def retrieve_documents(
    vector_store,
    question,
    k=TOP_K,
    fetch_k=20,
):
    """Retrieve relevant chunks from unique restaurants."""

    candidates = vector_store.similarity_search(
        question,
        k=fetch_k,
    )

    selected = []
    seen_restaurants = set()

    for document in candidates:

        place_id = document.metadata.get("place_id")

        if place_id in seen_restaurants:
            continue

        seen_restaurants.add(place_id)
        selected.append(document)

        if len(selected) == k:
            break

    return selected


def rank_restaurants(documents, question):
    """Rank restaurants according to the user's query."""

    query = question.lower()

    wants_cheap = any(
        word in query
        for word in [
            "cheap",
            "inexpensive",
            "affordable",
            "budget",
        ]
    )

    wants_wine = "wine" in query

    wants_delivery = any(
        word in query
        for word in [
            "delivery",
            "deliver",
        ]
    )

    def ranking_key(document):
        metadata = document.metadata

        rating = metadata.get("rating") or 0
        rating_count = (
            metadata.get("user_ratings_total") or 0
        )

        constraint_score = 0

        if wants_cheap:
            if metadata.get("price_level") == "inexpensive ($)":
                constraint_score += 1

        if wants_wine:
            if metadata.get("serves_wine") is True:
                constraint_score += 1

        if wants_delivery:
            if metadata.get("delivery") is True:
                constraint_score += 1

        return (
            constraint_score,
            rating,
            rating_count,
        )

    return sorted(
        documents,
        key=ranking_key,
        reverse=True,
    )

def build_context(documents):
    """Build structured context from retrieved restaurant chunks."""

    context_sections = []

    for document in documents:

        metadata = document.metadata

        name = metadata.get(
            "name",
            "Unknown restaurant",
        )

        rating = metadata.get(
            "rating",
            "Unknown",
        )

        rating_count = metadata.get(
            "user_ratings_total",
            "Unknown",
        )

        location = metadata.get(
            "location",
            "Unknown",
        )

        section = (
            f"Restaurant: {name}\n"
            f"Average rating: {rating}/5\n"
            f"Number of ratings: {rating_count}\n"
            f"Location: {location}\n\n"
            f"Retrieved information:\n"
            f"{document.page_content}"
        )

        context_sections.append(section)

    return "\n\n---\n\n".join(
        context_sections
    )


def build_prompt(
    tokenizer,
    question,
    context,
):
    """Build a grounded Phi-3 prompt for restaurant recommendations."""

    messages = [
        {
            "role": "system",
            "content": (
                "You are a restaurant recommendation assistant "
                "specialized in restaurants in Padova, Italy.\n\n"

                "You must answer using only the information provided "
                "in the restaurant information below.\n\n"

                "Rules:\n"
                "1. Do not invent facts that are not present in the "
                "provided information.\n"
                "2. Do not refer to documents, context, chunks, or "
                "document numbers in your answer.\n"
                "3. When the user asks for the best restaurant, do "
                "not treat a single positive review as proof that it "
                "is objectively the best.\n"
                "4. Compare restaurants using available evidence such "
                "as average rating, number of ratings, reviews, "
                "services, price level, and location.\n"
                "5. Give more importance to aggregate ratings and "
                "the number of ratings than to a single review.\n"
                "6. If the evidence is insufficient to identify one "
                "clear best option, say so and recommend the strongest "
                "candidate or candidates based on the available "
                "information.\n"
                "7. Explain briefly why the recommendation matches "
                "the user's request.\n"
                "8. Never claim that a restaurant offers a service, "
                "food, opening time, or other feature unless it "
                "appears in the provided information.\n"
                "9. Keep the answer concise: recommend at most two restaurants "
                "and answer in no more than 120 words."
            ),
        },
        {
            "role": "user",
            "content": (
                f"Restaurant information:\n\n"
                f"{context}\n\n"
                f"User question: {question}"
            ),
        },
    ]

    return tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )


class RestaurantRAG:
    """Restaurant recommendation RAG pipeline."""

    def __init__(self):

        print("Loading vector database...")

        self.vector_store = load_vector_store()

        print("Loading Phi-3...")

        self.tokenizer, self.generator = load_llm()

    def ask(
        self,
        question,
        k=TOP_K,
    ):
        """Answer a restaurant question using RAG."""

        # Retrieve semantically relevant restaurants
        documents = retrieve_documents(
            self.vector_store,
            question,
            k=k,
        )

        # Rank candidates using structured metadata
        documents = rank_restaurants(
            documents,
            question,
        )

        # Build structured context
        context = build_context(
            documents
        )

        # Build grounded prompt
        prompt = build_prompt(
            self.tokenizer,
            question,
            context,
        )

        # Generate answer
        result = self.generator(
            prompt
        )

        answer = (
            result[0]["generated_text"]
            .strip()
        )

        # Return answer and retrieved sources
        return {
            "question": question,
            "answer": answer,
            "sources": [
                {
                    "name": document.metadata.get(
                        "name"
                    ),
                    "place_id": document.metadata.get(
                        "place_id"
                    ),
                    "rating": document.metadata.get(
                        "rating"
                    ),
                    "user_ratings_total":
                        document.metadata.get(
                            "user_ratings_total"
                        ),
                    "chunk_id": document.metadata.get(
                        "chunk_id"
                    ),
                }
                for document in documents
            ],
        }


def main():
    """Run the Restaurant RAG command-line interface."""

    rag = RestaurantRAG()

    print("\nRestaurant RAG Padova")
    print("Type 'quit' to exit.\n")

    while True:

        question = input(
            "Question: "
        ).strip()

        if question.lower() in {
            "quit",
            "exit",
            "q",
        }:
            break

        if not question:
            continue

        result = rag.ask(
            question
        )

        print("\nAnswer:")
        print(
            result["answer"]
        )

        print(
            "\nRetrieved restaurants:"
        )

        for source in result["sources"]:

            rating = source.get(
                "rating"
            )

            rating_count = source.get(
                "user_ratings_total"
            )

            print(
                f"- {source['name']} "
                f"({rating}/5, "
                f"{rating_count} ratings)"
            )

        print()


if __name__ == "__main__":
    main()
