# Processed Data

The preprocessing pipeline cleans restaurant metadata and review data
before constructing the document corpus used by the RAG system.

The processed corpus is generated locally and is used to build the
Chroma vector database.

See:

- `src/preprocessing.py`
- `src/corpus.py`
- `src/vector_store.py`
