"""Create and load the Chroma vector store for Restaurant RAG Padova."""

import json
from pathlib import Path

import torch
from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter


# ---------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------

EMBEDDING_MODEL_NAME = "thenlper/gte-small"

CHUNK_SIZE = 500
CHUNK_OVERLAP = 50

DEFAULT_CORPUS_PATH = "data/processed/restaurants.json"
DEFAULT_DB_PATH = "chroma_db"


# ---------------------------------------------------------------------
# Embedding model
# ---------------------------------------------------------------------

def get_embedding_model():
    """Load the sentence embedding model."""

    device = "cuda" if torch.cuda.is_available() else "cpu"

    print(f"Loading embeddings on: {device}")

    return HuggingFaceEmbeddings(
        model_name=EMBEDDING_MODEL_NAME,
        model_kwargs={"device": device},
        encode_kwargs={
            "normalize_embeddings": True,
        },
    )


# ---------------------------------------------------------------------
# Corpus
# ---------------------------------------------------------------------

def load_corpus(corpus_path=DEFAULT_CORPUS_PATH):
    """Load the restaurant corpus from JSON."""

    path = Path(corpus_path)

    if not path.exists():
        raise FileNotFoundError(
            f"Corpus not found at {path}. "
            "Create the corpus before building the vector store."
        )

    with open(path, "r", encoding="utf-8") as file:
        return json.load(file)


# ---------------------------------------------------------------------
# Chunking
# ---------------------------------------------------------------------

def create_chunks(
    corpus,
    chunk_size=CHUNK_SIZE,
    chunk_overlap=CHUNK_OVERLAP,
):
    """Split restaurant documents into smaller retrieval chunks."""

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
    )

    documents = []

    for restaurant in corpus:

        text = restaurant["text"]
        metadata = restaurant["metadata"]

        # Include identifying information in every chunk.
        header = (
            f"Restaurant: {metadata.get('name', 'Unknown')}\n"
            f"Location: {metadata.get('location', 'Unknown')}\n"
            f"Rating: {metadata.get('rating', 'N/A')}\n"
        )

        chunks = splitter.split_text(text)

        for chunk_id, chunk in enumerate(chunks):

            chunk_metadata = metadata.copy()
            chunk_metadata["chunk_id"] = chunk_id

            # Chroma metadata should contain simple scalar values.
            chunk_metadata = {
                key: value
                for key, value in chunk_metadata.items()
                if value is not None
                and isinstance(
                    value,
                    (str, int, float, bool),
                )
            }

            documents.append(
                Document(
                    page_content=f"{header}\n{chunk}",
                    metadata=chunk_metadata,
                )
            )

    return documents


# ---------------------------------------------------------------------
# Vector database
# ---------------------------------------------------------------------

def build_vector_store(
    corpus_path=DEFAULT_CORPUS_PATH,
    persist_directory=DEFAULT_DB_PATH,
):
    """Create a persistent Chroma database from the restaurant corpus."""

    print("Loading corpus...")
    corpus = load_corpus(corpus_path)

    print(f"Loaded {len(corpus)} restaurant documents.")

    print("Creating chunks...")
    documents = create_chunks(corpus)

    print(f"Created {len(documents)} chunks.")

    embedding_model = get_embedding_model()

    print("Generating embeddings and building Chroma database...")

    vector_store = Chroma.from_documents(
        documents=documents,
        embedding=embedding_model,
        persist_directory=persist_directory,
        collection_metadata={
            "hnsw:space": "cosine",
        },
    )

    print(
        f"Vector database created at: {persist_directory}"
    )

    return vector_store


def load_vector_store(
    persist_directory=DEFAULT_DB_PATH,
):
    """Load an existing Chroma database."""

    path = Path(persist_directory)

    if not path.exists():
        raise FileNotFoundError(
            f"Vector database not found at {path}. "
            "Run this module first to create it."
        )

    embedding_model = get_embedding_model()

    return Chroma(
        persist_directory=persist_directory,
        embedding_function=embedding_model,
    )


# ---------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------

if __name__ == "__main__":
    build_vector_store()
