import json

from query import LegalRetriever

from evaluator import (
    calculate_recall,
    calculate_precision,
    calculate_mrr
)

retriever = LegalRetriever()

with open(
    "testset.json",
    "r",
    encoding="utf-8"
) as f:

    testset = json.load(f)

recall_scores = []
precision_scores = []
mrr_scores = []

print("\nRunning Evaluation\n")

for item in testset:

    question = item["question"]

    ground_truth_source = item["source"]

    result = retriever.retrieve(
        question,
        k=3
    )

    retrieved_sources = [
        meta["source"]
        for meta in result["metadatas"]
    ]

    recall = calculate_recall(
        retrieved_sources,
        ground_truth_source
    )

    precision = calculate_precision(
        retrieved_sources,
        ground_truth_source
    )

    mrr = calculate_mrr(
        retrieved_sources,
        ground_truth_source
    )

    recall_scores.append(recall)
    precision_scores.append(precision)
    mrr_scores.append(mrr)

    print("=" * 50)
    print("Question:", question)
    print("Ground Truth:", ground_truth_source)
    print("Retrieved:", retrieved_sources)
    print("Recall:", recall)
    print("Precision:", round(precision, 2))
    print("MRR:", round(mrr, 2))

avg_recall = sum(recall_scores) / len(recall_scores)

avg_precision = (
    sum(precision_scores)
    / len(precision_scores)
)

avg_mrr = sum(mrr_scores) / len(mrr_scores)

print("\n" + "=" * 50)

print("FINAL RESULTS")

print(
    f"Recall@3: {avg_recall:.2f}"
)

print(
    f"Precision@3: {avg_precision:.2f}"
)

print(
    f"MRR: {avg_mrr:.2f}"
)

print("=" * 50)