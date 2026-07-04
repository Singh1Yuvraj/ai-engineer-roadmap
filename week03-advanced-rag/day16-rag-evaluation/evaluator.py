def calculate_recall(retrieved, ground_truth):

    if ground_truth in retrieved:
        return 1

    return 0


def calculate_precision(retrieved, ground_truth):

    relevant = 0

    for item in retrieved:
        if item == ground_truth:
            relevant += 1

    return relevant / len(retrieved)


def calculate_mrr(retrieved, ground_truth):

    for rank, item in enumerate(retrieved, start=1):

        if item == ground_truth:
            return 1 / rank

    return 0