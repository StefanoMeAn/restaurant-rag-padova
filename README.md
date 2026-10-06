# Restaurant RAG Padova 

A Retrieval-Augmented Generation (RAG) project for searching and recommending restaurants in **Padova, Italy**.

I built this project as part of my NLP coursework during my Master's in Physics of Data at the University of Padova. The idea was to explore how semantic search and a local language model could be combined to answer natural-language questions about restaurants.

Instead of starting from an existing dataset, I collected the restaurant data myself using the Google Places API. The final dataset contains **556 restaurants and 2,545 reviews**.

Examples of questions the system can handle:

> *"Recommend a cheap restaurant."*

> *"Where can I find good vegan food?"*

> *"I want a restaurant with good reviews that serves wine."*

> *"Which restaurant is good for a romantic dinner?"*

> *"Recommend a restaurant that offers delivery."*

---

## How it works

The pipeline combines semantic retrieval with structured restaurant information:

```text
User query
    │
    ▼
GTE-small embeddings
    │
    ▼
Chroma semantic search
    │
    ▼
Candidate restaurants
    │
    ▼
Query-aware reranking
    │
    ▼
Top restaurants
    │
    ▼
Phi-3 Mini
    │
    ▼
Natural-language answer
```

The main components are:

- **GTE-small** for text embeddings
- **Chroma** as the vector database
- **LangChain** for the retrieval pipeline
- **Phi-3 Mini 4K Instruct** as the local language model
- **Google Places API** for building the original dataset

Phi-3 runs locally in **4-bit quantization with bitsandbytes**.

---

## Building the dataset

One part of the project I wanted to implement myself was the data collection.

I divided Padova into overlapping geographic search regions and used the Google Places API to discover restaurants. I then retrieved detailed information and available reviews for each place.

After cleaning and deduplication, the dataset contains:

| | Count |
|---|---:|
| Restaurants | **556** |
| Reviews | **2,545** |

For each restaurant I collected information such as:

- name and address
- rating and number of ratings
- price level
- restaurant type
- delivery and dine-in availability
- reservation availability
- breakfast/lunch/dinner service
- beer and wine availability
- opening hours
- coordinates
- customer reviews

The complete Google-derived dataset is not distributed in this repository, but the collection methodology is available in `src/data_collection.py`.

---

## Preparing the RAG corpus

After preprocessing the raw data, I construct one document for each restaurant combining structured metadata with its available reviews.

The documents are split using:

```text
chunk size    = 500
chunk overlap = 50
```

This produces approximately **2,587 chunks** from the 556 restaurant documents.

The chunks are embedded using:

```text
thenlper/gte-small
```

with normalized embeddings and stored in Chroma using cosine similarity.

---

## Improving the retrieval

The first version of the project used standard semantic retrieval.

While testing it, I noticed an important problem.

For a question such as:

> *"Recommend a cheap restaurant."*

semantic similarity could retrieve restaurants that were relevant to the question but were not actually classified as inexpensive.

The same problem appeared with constraints such as **delivery**.

So I changed the pipeline to combine semantic retrieval with the structured metadata already available in the dataset.

The current version works approximately like this:

```text
Semantic search
      ↓
20 unique restaurant candidates
      ↓
Detect structured query constraints
      ↓
Metadata-aware reranking
      ↓
Top 4 restaurants
```

At the moment I explicitly handle constraints related to:

- price
- wine availability
- delivery

This keeps semantic search useful for finding relevant candidates while using structured information when the user asks for something that can be checked directly.

---

## Evaluation

I compared three versions of the retrieval pipeline:

**Baseline** — semantic retrieval followed by rating-based ranking.

**V1** — query-aware reranking of the original four candidates.

**V2** — retrieve a larger pool of 20 candidates first, then apply query-aware reranking.

For three queries with constraints that can be checked directly from the metadata, I measured the percentage of the top four recommendations satisfying the requested constraint.

| Query | Baseline@4 | V1@4 | V2@4 |
|---|---:|---:|---:|
| Cheap | 25% | 25% | **100%** |
| Wine | 100% | 100% | **100%** |
| Delivery | 50% | 50% | **100%** |
| **Average** | **58.3%** | **58.3%** | **100%** |

For the first recommendation:

| Strategy | Constraint satisfaction @1 |
|---|---:|
| Baseline | 66.7% |
| V1 | 100% |
| V2 | 100% |

These numbers are deliberately limited to the **three structured-constraint queries** above. They are not meant to represent overall RAG accuracy.

The experiment mainly showed me that increasing the candidate pool is useful when a second ranking stage needs to enforce explicit constraints.

---

## What I learned

One of the more interesting parts of this project was seeing the difference between **retrieval quality and generation quality**.

After improving retrieval, the system could correctly identify restaurants satisfying structured constraints. However, Phi-3 could still occasionally introduce details that were not fully supported by the retrieved information.

For example, the model sometimes added descriptions about atmosphere, food quality, or other characteristics that were not explicitly present in the evidence.

This was a useful result because it showed that:

```text
good retrieval ≠ automatically grounded generation
```

For information already represented as structured metadata, a deterministic answer can provide stronger factual guarantees. A language model becomes more useful when the system needs to interpret unstructured information such as customer reviews.

This is one of the main directions I would explore in a future version of the project.

---

## Example

For:

```text
Recommend a restaurant that offers delivery.
```

the query-aware retrieval stage can return:

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

The important difference from the original retrieval approach is that **delivery is checked using structured metadata**, rather than relying only on similarity between the query and restaurant text.

---

## Project structure

```text
restaurant-rag-padova/
│
├── build_pipeline.py
├── README.md
├── requirements.txt
│
├── data/
│   ├── raw/
│   └── processed/
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

## Running the project

Clone the repository:

```bash
git clone git@github.com:StefanoMeAn/restaurant-rag-padova.git
cd restaurant-rag-padova
```

Create an environment and install the dependencies:

```bash
python -m venv .venv
source .venv/bin/activate

pip install -r requirements.txt
```

Build the preprocessing and vector database pipeline:

```bash
python build_pipeline.py
```

Run the RAG system:

```bash
python -m src.rag_pipeline
```

Run the retrieval evaluation:

```bash
python evaluation/compute_metrics.py
```

### GPU setup

I tested the local generation pipeline with an NVIDIA GPU using:

```text
PyTorch 2.7.1
CUDA 11.8
```

The exact PyTorch installation may need to be adapted to the local NVIDIA driver.

---

## Recollecting the data

The original dataset was collected using the Google Places API.

Create a local `.env` file:

```bash
cp .env.example .env
```

and set:

```text
GOOGLE_MAPS_API_KEY=your_api_key_here
```

API keys and the original raw dataset are excluded from Git.

---

## Limitations and future work

There are several things I would improve if I continued the project:

- expand query-intent detection beyond price, wine and delivery
- improve geographic queries using explicit distances
- evaluate semantic relevance on a larger manually labelled test set
- evaluate generation faithfulness separately from retrieval
- combine deterministic answers for structured facts with LLM-based review summarization
- experiment with reranking models instead of hand-written constraint rules

The restaurant data is also static, so this system should be considered an NLP/RAG experiment rather than a real-time restaurant recommendation service.

---

## Tech stack

`Python` · `PyTorch` · `Transformers` · `Sentence Transformers` · `LangChain` · `Chroma` · `Phi-3` · `GTE-small` · `pandas` · `Google Places API`

---

## Author

**Stefano Meza**

Physicist with a Master's degree in Physics of Data from the **University of Padova**, interested in machine learning, deep learning, computer vision, NLP and scientific computing.

[LinkedIn](https://www.linkedin.com/in/stefanomean/) · [GitHub](https://github.com/StefanoMeAn)
