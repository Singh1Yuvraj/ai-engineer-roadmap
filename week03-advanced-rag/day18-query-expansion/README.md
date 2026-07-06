# Day 18 – Query Expansion & Multi-Query Retrieval

## Objective

The goal of this project is to improve Retrieval-Augmented Generation (RAG) by expanding a user's query into multiple semantically related queries before performing retrieval.

Unlike traditional RAG, which searches using only the original question, this implementation generates multiple search queries, retrieves results for each, merges the retrieved chunks, removes duplicates, and returns a richer context for downstream LLMs.

---

# Concepts Covered

* Query Expansion
* Multi-Query Retrieval
* Vector Search using ChromaDB
* Sentence Embeddings
* Result Fusion
* Deduplication
* Recall
* Mean Reciprocal Rank (MRR)
* Latency Evaluation

---

# Project Structure

```text
day18-query-expansion/

├── data/
│   ├── nda.txt
│   ├── employment.txt
│   └── contract_termination.txt
│
├── chroma/
│
├── ingest.py
├── retriever.py
├── query_expansion.py
├── evaluator.py
├── main.py
├── requirements.txt
└── README.md
```

---

# Project Workflow

```text
Legal Documents
        │
        ▼
Recursive Chunking
        │
        ▼
Embeddings (all-MiniLM-L6-v2)
        │
        ▼
ChromaDB
        │
        ▼
User Question
        │
        ▼
Query Expansion
        │
        ▼
Multiple Queries
        │
        ▼
Vector Search
        │
        ▼
Result Fusion
        │
        ▼
Deduplication
        │
        ▼
Retrieved Context
```

---

# Query Expansion

Example

Original Question

```
Can employees disclose company secrets?
```

Expanded Queries

```
Can employees disclose company secrets?

Can workers disclose company secrets?

Can staff disclose company secrets?

Can employees disclose confidential information?

Can employees disclose trade secrets?
```

Each expanded query performs an independent vector search.

---

# Multi-Query Retrieval

Instead of retrieving documents once, the system retrieves documents for every expanded query.

Example

```
Query 1

A
B
C

Query 2

B
D
E

Query 3

A
F
G
```

After Result Fusion

```
A
B
C
B
D
E
A
F
G
```

After Deduplication

```
A
B
C
D
E
F
G
```

These unique chunks become the final retrieval context.

---

# Evaluation Metrics

## Recall

Measures whether the relevant information was successfully retrieved.

Formula

```
Relevant Retrieved
------------------
Total Relevant
```

Higher Recall means fewer relevant chunks are missed.

---

## Mean Reciprocal Rank (MRR)

Measures how highly the first relevant chunk is ranked.

Formula

```
MRR = Average(1 / Rank)
```

Higher MRR indicates that relevant chunks appear earlier in the results.

---

## Latency

Measures the average retrieval time for a query.

```
Latency = End Time − Start Time
```

Lower latency generally leads to a better user experience.

---

# Running the Project

## Step 1 – Build the Vector Database

```bash
python3 ingest.py
```

Expected Output

```
Stored 179 chunks.
```

---

## Step 2 – Test Retrieval

```bash
python3 main.py
```

Example Question

```
Can employees disclose company secrets?
```

The application displays:

* Single Query Retrieval
* Multi-Query Retrieval

---

## Step 3 – Evaluate Performance

```bash
python3 evaluator.py
```

Example Output

```
Evaluation Results

Single Query

Recall : 0.60
MRR    : 0.47
Latency: 1.82 ms

Multi Query

Recall : 0.80
MRR    : 0.66
Latency: 5.91 ms
```

---

# Learning Outcomes

By completing this project you will understand:

* Why users and documents often use different vocabulary
* How Query Expansion improves Recall
* How Multi-Query Retrieval works
* How Result Fusion combines multiple retrievals
* How Deduplication removes repeated chunks
* Why Multi-Query Retrieval generally returns richer context
* Trade-offs between retrieval quality and latency

---

# Future Improvements

* LLM-based Query Expansion
* Reciprocal Rank Fusion (RRF)
* Maximum Marginal Relevance (MMR)
* Cross-Encoder Re-ranking
* Hybrid Search (BM25 + Vector Search)
* HyDE Retrieval
* Self-RAG
* Citation-aware Retrieval

---

# Technologies Used

* Python
* ChromaDB
* Sentence Transformers
* LangChain Text Splitters
* all-MiniLM-L6-v2
* Pandas

---

# Key Takeaway

Traditional RAG performs retrieval using a single query.

Multi-Query Retrieval improves recall by generating multiple semantically related queries, retrieving documents for each query, merging the results, removing duplicates, and providing the LLM with richer context.

This approach reduces vocabulary mismatch and significantly improves retrieval quality for production RAG systems.
