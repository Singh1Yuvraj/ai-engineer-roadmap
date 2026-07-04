import time
import chromadb
import pandas as pd
from sentence_transformers import SentenceTransformer

model = SentenceTransformer(
    "all-MiniLM-L6-v2"
)

queries = [
    {
        "question":
        "What happens after employment termination?",

        "expected":
        "termination"
    },

    {
        "question":
        "How long does confidentiality survive?",

        "expected":
        "five years"
    },

    {
        "question":
        "Can employees disclose company secrets?",

        "expected":
        "trade secrets"
    }
]


def evaluate_collection(collection_name):

    client = chromadb.PersistentClient(
        path="./chroma"
    )

    collection = client.get_collection(
        collection_name
    )

    recall_hits = 0
    reciprocal_ranks = []
    latencies = []

    for q in queries:

        query_embedding = model.encode(
            q["question"]
        ).tolist()

        start = time.time()

        results = collection.query(
            query_embeddings=[query_embedding],
            n_results=5
        )

        latency = (
            time.time() - start
        ) * 1000

        latencies.append(latency)

        docs = results["documents"][0]

        found = False

        for rank, doc in enumerate(
            docs,
            start=1
        ):

            if q["expected"].lower() in doc.lower():

                recall_hits += 1

                reciprocal_ranks.append(
                    1 / rank
                )

                found = True

                break

        if not found:
            reciprocal_ranks.append(0)

    recall = (
        recall_hits /
        len(queries)
    )

    mrr = (
        sum(reciprocal_ranks) /
        len(queries)
    )

    avg_latency = (
        sum(latencies) /
        len(latencies)
    )

    return {
        "Collection": collection_name,
        "Recall": round(recall, 2),
        "MRR": round(mrr, 2),
        "Latency(ms)": round(
            avg_latency,
            2
        )
    }


collections = [
    "fixed_200",
    "fixed_500",
    "fixed_1000",
    "recursive_200",
    "recursive_500",
    "recursive_1000",
    "legal"
]

results = []

for collection in collections:
    results.append(
        evaluate_collection(collection)
    )

df = pd.DataFrame(results)

print("\n")
print(df)

df.to_csv(
    "results.csv",
    index=False
)

print(
    "\nResults saved to results.csv"
)