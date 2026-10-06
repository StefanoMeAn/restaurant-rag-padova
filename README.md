# Restaurant RAG Padova

A Retrieval-Augmented Generation (RAG) system for restaurant search and recommendation in **Padova, Italy**, built from a custom dataset collected using the Google Places API.

The project combines semantic search, structured restaurant metadata, query-aware ranking, and a local language model to answer natural-language restaurant queries.

## Overview

Restaurant recommendations often require combining different types of information: cuisine, price, services, ratings, location, and information contained in customer reviews.

This project explores a hybrid retrieval architecture that combines:

- restaurant data collected through the Google Places API;
- semantic retrieval using transformer embeddings;
- a Chroma vector database;
- structured metadata constraints;
- query-aware reranking;
- local answer generation with Phi-3.

The dataset contains **556 restaurants** and **2,545 reviews** from Padova.

Example queries include:

> "Recommend a cheap restaurant."

> "I want a restaurant with good reviews that serves wine."

> "Where can I find good vegan food?"

> "Which restaurant is good for a romantic dinner?"

> "Recommend a restaurant that offers delivery."

---

## System Architecture

```text
                    ┌──────────────────────┐
                    │     User Query       │
                    └──────────┬───────────┘
                               │
                               ▼
                    ┌──────────────────────┐
                    │   GTE-small Query    │
                    │      Embedding       │
                    └──────────┬───────────┘
                               │
                               ▼
                    ┌──────────────────────┐
                    │       Chroma         │
                    │  Semantic Retrieval  │
                    └──────────┬───────────┘
                               │
                               ▼
                    ┌──────────────────────┐
                    │ Candidate Restaurant │
                    │        Pool          │
                    └──────────┬───────────┘
                               │
                               ▼
                    ┌──────────────────────┐
                    │ Query-Aware Ranking  │
                    │ + Metadata Filters   │
                    └──────────┬───────────┘
                               │
                               ▼
                    ┌──────────────────────┐
                    │ Top Restaurants      │
                    └──────────┬───────────┘
                               │
                               ▼
                    ┌──────────────────────┐
                    │ Phi-3-mini-4k        │
                    │ Local Generation     │
                    └──────────┬───────────┘
                               │
                               ▼
                    ┌──────────────────────┐
                    │ Recommendation       │
                    └──────────────────────┘
```

---

## Dataset

The dataset was created specifically for this project using the **Google Places API**.

Padova was divided into overlapping geographic search regions. Restaurants were discovered using nearby searches and enriched using place details.

The resulting local dataset contains:

| Data | Count |
|---|---:|
| Restaurants | 556 |
| Reviews | 2,545 |

Restaurant metadata includes information such as:

- name and address;
- average rating and number of ratings;
- price level;
- restaurant type;
- delivery and dine-in availability;
- reservation availability;
- breakfast, lunch, and dinner service;
- beer and wine availability;
- opening hours;
- geographic coordinates.

Customer reviews are linked to restaurants using their `place_id`.

The complete original Google-derived dataset is not distributed through this repository. The data-collection methodology is available in `src/data_collection.py`.

---

## RAG Pipeline

### 1. Data preprocessing

Raw restaurant metadata and reviews are cleaned and transformed into a consistent representation.

The preprocessing pipeline handles:

- restaurant categories;
- price levels;
- opening hours;
- geographic coordinates;
- service information;
- review text.

### 2. Document construction

A textual document is generated for each restaurant by combining its metadata with available customer reviews.

Structured metadata is preserved alongside the text for later filtering and ranking.

### 3. Chunking

Restaurant documents are divided using a recursive text splitter with:

```text
Chunk size:    500 characters
Chunk overlap: 50 characters
```

The current corpus produces approximately **2,587 chunks** from 556 restaurant documents.

### 4. Embeddings

Chunks are embedded using:

**`thenlper/gte-small`**

Embeddings are L2-normalized before storage.

### 5. Vector database

The embeddings are stored in **Chroma** using cosine similarity.

At query time, the system performs semantic retrieval to identify restaurant candidates relevant to the user's request.

---

## Query-Aware Retrieval

A semantic search alone does not guarantee that structured user constraints are satisfied.

For example:

> "Recommend a cheap restaurant."

A semantically relevant restaurant may still have a moderate or expensive price level.

The retrieval pipeline was therefore extended with query-aware ranking.

The current system retrieves a larger pool of unique restaurant candidates and detects structured constraints in the query. Relevant metadata is then used during reranking.

Currently implemented constraints include:

- inexpensive restaurants;
- wine availability;
- delivery availability.

The final ranking also considers restaurant ratings and the number of user ratings.

This produces the following pipeline:

```text
Semantic retrieval
        ↓
20 unique restaurant candidates
        ↓
Query constraint detection
        ↓
Metadata-aware reranking
        ↓
Top 4 restaurants
```

---

## Evaluation

Retrieval was evaluated incrementally using three strategies.

**Baseline**

Semantic retrieval followed by rating-based ranking.

**V1 — Query-aware reranking**

Structured constraints are used to rerank the original four retrieved restaurants.

**V2 — Larger candidate pool**

Twenty unique semantic candidates are retrieved before applying query-aware reranking.

### Structured-constraint evaluation

Three queries with directly measurable metadata constraints were evaluated:

- cheap restaurant;
- serves wine;
- offers delivery.

Constraint satisfaction among the top four results:

| Query | Baseline@4 | V1@4 | V2@4 |
|---|---:|---:|---:|
| Cheap | 25% | 25% | **100%** |
| Wine | 100% | 100% | **100%** |
| Delivery | 50% | 50% | **100%** |
| **Average** | **58.3%** | **58.3%** | **100%** |

Top-result constraint satisfaction:

| Strategy | Average @1 |
|---|---:|
| Baseline | 66.7% |
| Query-aware V1 | 100% |
| Query-aware V2 | 100% |

These metrics measure **constraint satisfaction on this three-query structured evaluation subset**. They should not be interpreted as general retrieval accuracy.

The experiments illustrate an important distinction: semantic similarity is useful for discovering relevant candidates, while structured metadata is more reliable for enforcing explicit constraints.

---

## Local Language Model

Answer generation uses:

**Microsoft Phi-3 Mini 4K Instruct**

The model is loaded locally using 4-bit quantization with `bitsandbytes`, allowing the complete RAG pipeline to run on consumer hardware.

The model receives the ranked restaurants and retrieved evidence and generates a concise natural-language response.

During evaluation, the model occasionally introduced unsupported descriptive claims despite receiving grounded context. This highlights a common RAG limitation: successful retrieval does not guarantee fully faithful generation.

For structured queries, deterministic use of metadata can provide stronger factual guarantees, while the language model remains useful for interpreting and summarizing unstructured review information.

---

## Example

Query:

```text
Recommend a restaurant that offers delivery.
```

The query-aware retrieval system returns candidates such as:

```text
1. XIANG DIMSUM
   Rating: 4.9/5
   Ratings: 452
   Delivery: True

2. TAD-K Take A. & Delivery
   Rating: 4.8/5
   Ratings: 52
   Delivery: True

3. Wok Time
   Rating: 4.7/5
   Delivery: True
```

Structured metadata ensures that delivery availability is considered explicitly rather than inferred only from semantic similarity.

---

## Project Structure

```text
restaurant-rag-padova/
│
├── build_pipeline.py
├── README.md
├── requirements.txt
├── .env.example
│
├── data/
│   ├── raw/
│   │   └── README.md
│   └── processed/
│       └── README.md
│
├── src/
│   ├── data_collection.py
│   ├── preprocessing.py
│   ├── corpus.py
│   ├── vector_store.py
│   └── rag_pipeline.py
│
├── evaluation/
│   ├── questions.json
│   ├── evaluate_retrieval.py
│   ├── compute_metrics.py
│   └── results/
│
├── notebooks/
└── assets/
```

---

## Installation

Clone the repository:

```bash
git clone git@github.com:StefanoMeAn/restaurant-rag-padova.git
cd restaurant-rag-padova
```

Create a virtual environment:

```bash
python -m venv .venv
source .venv/bin/activate
```

Install the dependencies:

```bash
pip install -r requirements.txt
```

### GPU support

The project was tested with an NVIDIA GPU using CUDA-enabled PyTorch.

The tested environment used:

```text
PyTorch 2.7.1
CUDA 11.8
```

A CUDA-compatible PyTorch installation can be installed separately depending on the local NVIDIA driver and CUDA environment.

---

## Google Places API

To recollect restaurant data, create a `.env` file:

```bash
cp .env.example .env
```

and add your API key:

```text
GOOGLE_MAPS_API_KEY=your_api_key_here
```

API keys must never be committed to the repository.

The project can also be explored using an already prepared local dataset without recollecting data from the API.

---

## Building the Pipeline

The preprocessing, corpus construction, and vector database can be built with:

```bash
python build_pipeline.py
```

The resulting Chroma database is stored locally and is excluded from Git.

---

## Running the RAG System

Run:

```bash
python -m src.rag_pipeline
```

The system loads the embedding model, Chroma database, and local Phi-3 model before accepting restaurant queries.

---

## Retrieval Evaluation

Run the retrieval experiments with:

```bash
python evaluation/evaluate_retrieval.py
```

Compare the baseline and query-aware strategies with:

```bash
python evaluation/compute_metrics.py
```

---

## Limitations

This project is an experimental RAG system rather than a production restaurant recommendation service.

Current limitations include:

- the restaurant dataset is static and may become outdated;
- structured query detection currently supports a limited set of constraints;
- subjective queries such as "romantic" or "family-friendly" depend primarily on semantic evidence from reviews;
- the local Phi-3 model can occasionally generate unsupported descriptive statements;
- retrieval constraint metrics cover a small structured evaluation subset and do not measure overall recommendation quality;
- restaurant availability and opening status are not retrieved in real time;
- geographic queries could be improved using explicit distance calculations.

These limitations provide directions for future work, including hybrid deterministic/generative answers, broader intent detection, reranking models, geographic filtering, and more extensive retrieval and generation evaluation.

---

## Technologies

**Language**

- Python

**Machine Learning / NLP**

- PyTorch
- Hugging Face Transformers
- Sentence Transformers
- Phi-3 Mini
- GTE-small
- bitsandbytes

**RAG / Retrieval**

- LangChain
- Chroma
- cosine similarity

**Data**

- pandas
- Google Places API
- BeautifulSoup
- geopy

---

## Author

**Stefano Meza**

Physicist with a Master's degree in Physics of Data from the University of Padova, with experience in machine learning, deep learning, computer vision, NLP, and scientific computing.

- LinkedIn: [stefanomean](https://www.linkedin.com/in/stefanomean/)
- GitHub: [StefanoMeAn](https://github.com/StefanoMeAn)

---

## License

This repository contains source code and evaluation material for educational and portfolio purposes.

The original Google Places data is not distributed with the repository.
