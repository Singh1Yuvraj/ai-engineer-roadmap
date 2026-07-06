import time

from retriever import LegalRetriever

queries = [

    {
        "question":
        "Can employees disclose company secrets?",

        "expected":
        "trade secrets"
    },

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
        "What law governs the agreement?",

        "expected":
        "India"
    },

    {
        "question":
        "When can a contract be terminated?",

        "expected":
        "material breach"
    }

]


retriever = LegalRetriever()


def evaluate(search_function):

    recall_hits = 0

    reciprocal_ranks = []

    latencies = []

    for item in queries:

        question = item["question"]

        expected = item["expected"]

        start = time.time()

        documents = search_function(question)

        latency = (
            time.time() - start
        ) * 1000

        latencies.append(latency)

        found = False

        for rank, doc in enumerate(
            documents,
            start=1
        ):

            if expected.lower() in doc.lower():

                recall_hits += 1

                reciprocal_ranks.append(
                    1 / rank
                )

                found = True

                break

        if not found:

            reciprocal_ranks.append(0)

    recall = recall_hits / len(queries)

    mrr = sum(reciprocal_ranks) / len(queries)

    avg_latency = sum(latencies) / len(latencies)

    return {

        "Recall": round(recall, 2),

        "MRR": round(mrr, 2),

        "Latency": round(avg_latency, 2)

    }


single = evaluate(
    retriever.single_query_search
)

multi = evaluate(
    retriever.multi_query_search
)

print("\n")
print("=" * 70)
print("Evaluation Results")
print("=" * 70)

print(
    f"""
Single Query

Recall : {single['Recall']}
MRR    : {single['MRR']}
Latency: {single['Latency']} ms
"""
)

print(
    f"""
Multi Query

Recall : {multi['Recall']}
MRR    : {multi['MRR']}
Latency: {multi['Latency']} ms
"""
)

print("=" * 70)

if multi["Recall"] > single["Recall"]:

    print("\nMulti-Query Retrieval improved Recall.")

elif multi["Recall"] == single["Recall"]:

    print("\nRecall remained the same.")

else:

    print("\nSingle Query performed better.")