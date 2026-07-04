# Day 17 - Advanced Chunking

## Goal

Compare different chunking strategies for Legal AI RAG systems.

---

## Chunking Strategies

### Fixed Chunking

Chunk Sizes:

- 200
- 500
- 1000

### Recursive Chunking

Uses LangChain RecursiveCharacterTextSplitter.

### Legal Chunking

Splits documents by legal clauses and sections.

---

## Metrics

### Recall

Measures whether the relevant chunk is retrieved.

Formula:

Recall = Relevant Retrieved / Total Queries

---

### MRR

Mean Reciprocal Rank

Formula:

MRR = Average(1 / Rank)

---

### Latency

Average query response time in milliseconds.

---

## Run

Build collections:

```bash
python chunking_experiments.py
```