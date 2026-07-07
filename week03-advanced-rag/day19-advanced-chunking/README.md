# Day 19 – Advanced Chunking Strategies for Legal RAG

## Overview

This project explores different chunking strategies used in Retrieval-Augmented Generation (RAG) systems for legal documents.

Instead of using a single chunking technique, this project evaluates multiple approaches and compares their impact on semantic retrieval quality.

The goal is to understand how chunking affects embeddings, vector search, and overall retrieval performance.

---

# Project Structure

```text
day19-advanced-chunking/
│
├── data/
│   ├── nda.txt
│   ├── employment.txt
│   └── contract_termination.txt
│
├── chunkers.py
├── embeddings.py
├── vector_store.py
├── retrieval.py
├── compare.py
├── main.py
│
├── chroma_db/
│
├── requirements.txt
└── README.md
```

---

# Architecture

```text
Legal Documents
        │
        ▼
Chunking Strategy
        │
        ▼
Text Chunks
        │
        ▼
Embedding Model
        │
        ▼
Dense Vectors
        │
        ▼
ChromaDB
        │
        ▼
Semantic Retrieval
        │
        ▼
Retrieved Legal Chunks
```

---

# Chunking Strategies Implemented

## 1. Fixed Chunking

* Splits documents using a fixed chunk size.
* Simple implementation.
* May break semantic boundaries.

### Best For

* Generic documents
* Simple RAG pipelines

---

## 2. Recursive Chunking

* Attempts to preserve paragraphs and sentences.
* Produces more meaningful chunks.
* Better semantic context.

### Best For

* Long-form documents
* Contracts
* Legal agreements

---

## 3. Sliding Window Chunking

* Uses overlapping chunks.
* Preserves context across chunk boundaries.

### Best For

* Question Answering
* Long contextual passages

---

## 4. Legal Section Chunking

* Splits documents based on legal headings.
* Keeps clauses together whenever possible.

### Best For

* Contracts
* NDAs
* Employment Agreements
* Policies

---

# Components

## chunkers.py

Responsible for:

* Fixed Chunking
* Recursive Chunking
* Sliding Window Chunking
* Legal Section Chunking

---

## embeddings.py

Responsible for:

* Loading SentenceTransformer
* Generating document embeddings
* Generating query embeddings

Model Used

```
all-MiniLM-L6-v2
```

Embedding Dimension

```
384
```

---

## vector_store.py

Responsible for:

* Creating ChromaDB collections
* Storing vectors
* Similarity search
* Collection management

---

## retrieval.py

Responsible for:

* Query embedding
* Semantic search
* Metadata filtering
* Pretty printing retrieval results

---

## compare.py

Responsible for evaluating all chunking strategies.

Comparison Metrics

* Number of Chunks
* Average Chunk Size
* Minimum Chunk Size
* Maximum Chunk Size
* Retrieval Distance
* Processing Time

---

## main.py

Provides an interactive CLI.

Features

* Select chunking strategy
* Index legal documents
* Ask legal questions
* Compare all chunking strategies

---

# Sample Evaluation

```
Query:
Can employer terminate employee without notice?

Winner

Recursive Character Chunker
```

```
Query:
What constitutes a breach of the NDA?

Winner

Fixed Size Chunker
```

```
Query:
Post-Termination Obligations

Winner

Sliding Window Chunker
```

---

# Key Learnings

During this project I learned:

* How chunking directly impacts retrieval quality.
* Why different chunking strategies perform differently.
* How Sentence Transformers generate dense embeddings.
* How ChromaDB stores and retrieves vectors.
* How metadata improves document retrieval.
* How retrieval pipelines are structured in production RAG systems.
* How to build an evaluation framework for comparing chunking strategies.

---

# Technologies Used

* Python
* Sentence Transformers
* ChromaDB
* NumPy
* Logging
* Object-Oriented Programming

---

# Future Improvements

* Token-based chunking
* Hierarchical chunking
* Parent-Child Retrieval
* Hybrid Search (BM25 + Dense Retrieval)
* Maximal Marginal Relevance (MMR)
* Cross-Encoder Re-ranking
* Contextual Retrieval
* Retrieval Evaluation Metrics (Recall@K, MRR, NDCG)

---

# Skills Practiced

* Advanced Chunking
* Vector Embeddings
* Semantic Search
* ChromaDB
* Retrieval Pipelines
* Evaluation Framework Design
* Software Architecture
* Object-Oriented Design

---

# Day 19 Outcome

By the end of this project I can:

* Build multiple chunking strategies from scratch.
* Generate dense embeddings using Sentence Transformers.
* Store vectors in ChromaDB.
* Retrieve relevant legal documents using semantic search.
* Compare chunking strategies using an evaluation framework.
* Understand the impact of chunking on Retrieval-Augmented Generation systems.

This project serves as the foundation for the next stage of the roadmap, where advanced retrieval techniques such as Hybrid Search, MMR, Cross-Encoder Re-ranking, and Agentic Retrieval will be implemented.
