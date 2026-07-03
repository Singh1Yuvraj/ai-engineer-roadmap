# Day 15 - Advanced Retrieval Pipeline

## Objective

Build a production-style retrieval pipeline by combining:

* Query Expansion
* Hybrid Search (BM25 + Vector Search)
* Reciprocal Rank Fusion (RRF)
* Cross-Encoder Re-ranking
* Metadata Filtering
* Retrieval Evaluation

This improves both retrieval recall and ranking quality compared to a simple vector search system.



## Architecture

User Query
↓
Query Expansion
↓
Hybrid Search
├── BM25 Search
└── Vector Search
↓
Reciprocal Rank Fusion (RRF)
↓
Deduplication
↓
Metadata Filtering
↓
Cross Encoder Re-ranking
↓
Top Relevant Chunks



## Project Structure

day15-advanced-retrieval/

├── data/
│   ├── nda.txt
│   ├── employment.txt
│   └── vendor_agreement.txt
│
├── chroma_db/
│
├── ingest.py
├── embedding.py
├── bm25_search.py
├── vector_search.py
├── hybrid_search.py
├── reranker.py
├── query_expansion.py
├── metadata_filter.py
├── evaluation.py
├── main.py
│
├── README.md
├── TECHNICAL_REPORT.md
└── requirements.txt

---

## Features

### Query Expansion

Expands user queries using predefined legal synonyms.

Example:

termination clause

becomes

* contract termination
* termination clause
* cancellation provision
* exit clause

This improves retrieval recall.

---

### BM25 Search

Keyword-based retrieval using Rank-BM25.

Advantages:

* Exact term matching
* Strong lexical retrieval
* Fast search

---

### Vector Search

Semantic retrieval using:

sentence-transformers/all-MiniLM-L6-v2

Advantages:

* Captures meaning
* Handles paraphrases
* Finds semantically similar chunks

---

### Hybrid Search

Combines:

* BM25 Search
* Vector Search

using:

Reciprocal Rank Fusion (RRF)

Formula:

RRF Score = 1 / (K + Rank)

This balances lexical and semantic retrieval.

---

### Cross Encoder Re-ranking

Model:

cross-encoder/ms-marco-MiniLM-L-6-v2

Each query-document pair is scored directly.

Example:

(Query, Chunk)

↓

Cross Encoder

↓

Relevance Score

This significantly improves ranking quality.

---

### Metadata Filtering

Supports filtering retrieved chunks by metadata.

Examples:

* NDA documents
* Employment agreements
* Vendor agreements

Useful for enterprise and legal AI systems.

---

## Installation

Create virtual environment:

python3 -m venv venv

Activate:

source venv/bin/activate

Install dependencies:

pip install -r requirements.txt

---

## Build Vector Database

Run:

python ingest.py

Expected Output:

Loaded documents
Created chunks
Stored chunks in ChromaDB
Saved chunks.json

---

## Run Retrieval System

python main.py

Example Query:

termination clause

---

## Example Output

Expanded Queries:

* termination clause
* contract termination
* cancellation provision
* exit clause

Retrieved Chunks:

* employment.txt
* vendor_agreement.txt
* nda.txt

Final Ranked Results:

1. employment.txt
2. vendor_agreement.txt
3. nda.txt

---

## Learning Outcomes

After completing Day 15, I understand:

* Query Expansion
* Hybrid Retrieval
* BM25 Search
* Vector Search
* Reciprocal Rank Fusion
* Cross Encoder Re-ranking
* Metadata Filtering
* Retrieval Evaluation

These concepts are widely used in production-grade Legal AI and RAG systems.

---

## Next Step

Day 16:

Context Compression and Retrieval Optimization

Goal:

Reduce irrelevant context before sending information to an LLM while maintaining answer quality.
