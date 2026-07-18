# Day 21: Context Compression RAG Pipeline

A production-grade post-retrieval optimization subsystem engineered to maximize context window utilization, minimize LLM inference costs, and strip away peripheral legal boilerplate text before generation.

## Subsystem Architecture
* **`chunkers.py`**: Handles sliding-window structural text fragmentation.
* **`parent_child.py`**: Maps granular high-precision child segments back to broad, informational parent blocks.
* **`metadata_filter.py`**: Implements composite pre-filtering layers (clauses, risk thresholds, date parameters).
* **`compressor.py`**: Executes advanced sentence token pruning and keyword density isolation algorithms.
* **`token_budget.py`**: Monitors character-to-token allocations and handles structural truncation boundaries.
* **`retrieval.py`**: The main orchestration gateway managing search strategies.
* **`compare.py`**: An analytics layer tracking token reduction percentages and context compression ratios.

## Quick Start
To launch the interactive CLI sandbox, run the execution script directly from this directory:

```bash
# Clean historical artifacts
rm -rf chroma_db/

# Boot up the interactive CLI runner
python3 main.py   