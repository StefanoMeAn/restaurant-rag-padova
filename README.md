# Restaurant Recommendation System with RAG

This project implements a Retrieval-Augmented Generation (RAG) system for searching and recommending restaurants in Padova, Italy. It combines semantic search over restaurant descriptions and reviews with structured metadata, then uses a local language model to generate natural-language answers.

The dataset contains **556 restaurants and 2,545 reviews**, collected using the Google Places API. In a small evaluation focused on explicit metadata constraints, query-aware reranking increased average top-four constraint satisfaction from **58.3% to 100%** across three test queries.

## Project Overview

This project was developed as part of NLP coursework during the Master’s in Physics of Data at the University of Padova. It examines how semantic retrieval and a local language model work together for restaurant search, and whether structured restaurant attributes improve recommendations when a query includes a checkable requirement.

Example queries include:

- “Recommend a cheap restaurant.”
- “Where can I find good vegan food?”
- “I want a restaurant with good reviews that serves wine.”
- “Which restaurant is good for a romantic dinner?”
- “Recommend a restaurant that offers delivery.”

## Dataset

Restaurant data were collected using the Google Places API. Padova was divided into overlapping geographic search regions; restaurant details and reviews were then retrieved, cleaned, and deduplicated.

| Item | Count |
|---|---:|
| Restaurants | 556 |
| Reviews | 2,545 |
| Corpus chunks | Approximately 2,587 |

The collected fields include name, address, rating, number of ratings, price level, restaurant type, delivery and dine-in availability, reservation availability, meal services, beer and wine availability, opening hours, coordinates, and customer reviews.

The Google-derived dataset and API credentials are not included in the repository. The collection procedure is documented in `src/data_collection.py`.

## Methodology

### Retrieval and generation pipeline

```text
User query
    ↓
GTE-small embeddings
    ↓
Chroma semantic search
    ↓
Candidate restaurants
    ↓
Query-aware reranking
    ↓
Top restaurants
    ↓
Local Phi-3 Mini
    ↓
Natural-language answer
```

The main components are:

- **Google Places API** for collecting the source data.
- **thenlper/gte-small** for normalized text embeddings.
- **Chroma** for cosine-similarity search over the corpus.
- **LangChain** for assembling the retrieval pipeline.
- **Phi-3 Mini 4K Instruct** as the local language model, run in 4-bit quantization with `bitsandbytes`.

### Corpus preparation

After preprocessing, the pipeline creates one document per restaurant by combining its structured information with available reviews. The documents are split into chunks of **500 characters** with **50 characters** of overlap, producing approximately 2,587 chunks. The chunks are embedded and stored in Chroma.

### Query-aware reranking

The initial version used semantic retrieval followed by rating-based ranking. During testing, semantic similarity alone sometimes returned restaurants that were relevant to a query but did not meet explicit requirements, such as being inexpensive or offering delivery.

The current version retrieves **20 unique restaurant candidates**, detects supported constraints, and reranks candidates using restaurant metadata before returning the top four. Explicitly handled constraints currently include **price, wine availability, and delivery**. Semantic retrieval remains useful for finding candidates, while metadata is used to check requirements that can be verified directly.

## Results

### Retrieval evaluation

Three retrieval strategies were compared on three queries whose constraints can be checked directly against the metadata:

- **Baseline:** semantic retrieval followed by rating-based ranking.
- **V1:** query-aware reranking of the original four candidates.
- **V2:** retrieval of 20 candidates followed by query-aware reranking.

The metric is the percentage of the top four recommendations satisfying the requested constraint.

| Query | Baseline@4 | V1@4 | V2@4 |
|---|---:|---:|---:|
| Cheap | 25% | 25% | **100%** |
| Wine | 100% | 100% | **100%** |
| Delivery | 50% | 50% | **100%** |
| **Average** | **58.3%** | **58.3%** | **100%** |

For the first recommendation, constraint satisfaction was **66.7%** for the baseline and **100%** for both query-aware strategies. These results apply only to the three structured-constraint queries; they are not a measure of overall RAG accuracy. The experiment suggests that retrieving a larger candidate pool helps a second ranking stage enforce explicit constraints.

### Example recommendation

For the query “Recommend a restaurant that offers delivery,” the query-aware retrieval stage can return:

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

Here, delivery is checked using structured metadata rather than inferred only from text similarity.

### Observations

The results show that retrieval quality and generation quality are separate. After retrieval was improved, the system could identify restaurants satisfying structured constraints, but Phi-3 could still add details about atmosphere or food quality that were not clearly supported by the retrieved evidence.

For structured facts, deterministic answers can provide stronger factual guarantees. A language model is more useful when interpreting unstructured information such as customer reviews. This distinction is an important direction for a future version.

## Repository Structure

```text
restaurant-rag-padova/
├── build_pipeline.py
├── README.md
├── requirements.txt
├── data/
│   ├── raw/
│   └── processed/
├── src/
│   ├── data_collection.py
│   ├── preprocessing.py
│   ├── corpus.py
│   ├── vector_store.py
│   └── rag_pipeline.py
├── evaluation/
│   ├── questions.json
│   ├── evaluate_retrieval.py
│   ├── compute_metrics.py
│   └── results/
├── notebooks/
└── assets/
```

The raw Google Places data and API credentials are excluded from Git.

## Installation

Clone the repository and create a virtual environment:

```bash
git clone https://github.com/StefanoMeAn/restaurant-rag-padova.git
cd restaurant-rag-padova
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

The local generation pipeline was tested with **PyTorch 2.7.1** and **CUDA 11.8** on an NVIDIA GPU. Choose a PyTorch build compatible with the local driver and hardware. The `requirements.txt` file records Python dependencies; running the Phi-3 reader also requires downloading model weights from Hugging Face and enough CPU memory or GPU memory. The raw restaurant data and built vector database are not distributed with this repository, so new installations cannot run the full pipeline without first obtaining source data.

## Usage

The repository does not include the original Google Places dataset or the persisted Chroma database. To reconstruct the pipeline, obtain the source data first. You can collect a new dataset with a Google Places API key, subject to the API's availability, access requirements, and billing, or supply your own CSV files matching the expected schema.

To collect new data, copy `.env.example` to `.env`, set `GOOGLE_MAPS_API_KEY`, and run:

```bash
cp .env.example .env
python -m src.data_collection
```

The collection code saves `data/raw/restaurants.csv` and `data/raw/reviews.csv`. With those files present, build the preprocessing and vector database pipeline:

```bash
python build_pipeline.py
```

Run the RAG system after the vector database exists:

```bash
python -m src.rag_pipeline
```

Run the retrieval evaluation from the repository root:

```bash
python -m evaluation.compute_metrics
```

This script also requires the Chroma database. Keep API keys and the raw Google Places data out of Git. The stored evaluation summaries document the original experiments, but reproducing those scores requires the corresponding dataset and embedding store.

## Limitations and Future Work

- Query-intent detection currently handles price, wine, and delivery constraints; other intents need to be added.
- Geographic queries could use explicit distance calculations.
- The retrieval evaluation uses only three manually selected structured-constraint queries. A larger, manually labelled test set would give a stronger relevance evaluation.
- Generation faithfulness should be evaluated separately from retrieval quality.
- Structured facts could be answered deterministically, with the language model used for review summarization and other unstructured questions.
- Learned reranking models could be compared with the current hand-written constraint rules.
- The restaurant data is static, so this is an NLP/RAG experiment rather than a real-time recommendation service.

## Technologies

Python · PyTorch · Transformers · Sentence Transformers · LangChain · Chroma · Phi-3 · GTE-small · pandas · Google Places API

## Author

**Stefano Meza** — Physicist with a Master’s degree in Physics of Data from the University of Padova, interested in machine learning, deep learning, computer vision, NLP, and scientific computing.

[LinkedIn](https://www.linkedin.com/in/stefanomean/) · [GitHub](https://github.com/StefanoMeAn)
