"""RAG pipeline for restaurant recommendations in Padova."""

import torch
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    pipeline,
)

from vector_store import load_vector_store


# ---------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------

READER_MODEL_NAME = "microsoft/Phi-3-mini-4k-instruct"

TOP_K = 4


# ---------------------------------------------------------------------
# Language model
# ---------------------------------------------------------------------

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
        max_new_tokens=200,
        temperature=0.3,
        top_p=0.8,
        repetition_penalty=1.1,
        return_full_text=False,
    )

    return tokenizer, generator


# ---------------------------------------------------------------------
# Retrieval
# ---------------------------------------------------------------------

def retrieve_documents(
    vector_store,
    question,
    k=TOP_K,
):
    """Retrieve the most relevant restaurant chunks."""

    return vector_store.similarity_search(
        question,
        k=k,
    )


def build_context(documents):
    """Combine retrieved chunks into the LLM context."""

    context_sections = []

    for index, document in enumerate(documents, start=1):

        restaurant = document.metadata.get(
            "name",
            "Unknown restaurant",
        )

        section = (
            f"Document {index}\n"
            f"Restaurant: {restaurant}\n"
            f"{document.page_content}"
        )

        context_sections.append(section)

    return "\n\n".join(context_sections)


# ---------------------------------------------------------------------
# Prompt
# ---------------------------------------------------------------------

def build_prompt(tokenizer, question, context):
    """Build the Phi-3 chat prompt."""

    messages = [
        {
            "role": "system",
            "content": (
                "You are a restaurant recommendation assistant "
                "for Padova, Italy. "
                "Answer using only the information contained "
                "in the retrieved context. "
                "Do not invent restaurant information. "
                "If the context does not contain enough "
                "information to answer the question, say so. "
                "When recommending restaurants, explain briefly "
                "why they match the user's request. "
                "Keep the answer concise."
            ),
        },
        {
            "role": "user",
            "content": (
                f"Context:\n{context}\n\n"
                f"Question: {question}"
            ),
        },
    ]

    return tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )


# ---------------------------------------------------------------------
# RAG
# ---------------------------------------------------------------------

class RestaurantRAG:
    """Restaurant recommendation RAG pipeline."""

    def __init__(self):

        print("Loading vector database...")
        self.vector_store = load_vector_store()

        print("Loading Phi-3...")
        self.tokenizer, self.generator = load_llm()

    def ask(self, question, k=TOP_K):
        """Answer a restaurant question using RAG."""

        documents = retrieve_documents(
            self.vector_store,
            question,
            k=k,
        )

        context = build_context(documents)

        prompt = build_prompt(
            self.tokenizer,
            question,
            context,
        )

        result = self.generator(prompt)

        answer = result[0]["generated_text"].strip()

        return {
            "question": question,
            "answer": answer,
            "sources": [
                {
                    "name": document.metadata.get("name"),
                    "place_id": document.metadata.get("place_id"),
                    "chunk_id": document.metadata.get("chunk_id"),
                }
                for document in documents
            ],
        }


# ---------------------------------------------------------------------
# Interactive demo
# ---------------------------------------------------------------------

def main():

    rag = RestaurantRAG()

    print("\nRestaurant RAG Padova")
    print("Type 'quit' to exit.\n")

    while True:

        question = input("Question: ").strip()

        if question.lower() in {
            "quit",
            "exit",
            "q",
        }:
            break

        if not question:
            continue

        result = rag.ask(question)

        print("\nAnswer:")
        print(result["answer"])

        print("\nRetrieved restaurants:")

        for source in result["sources"]:
            print(f"- {source['name']}")

        print()


if __name__ == "__main__":
    main()
