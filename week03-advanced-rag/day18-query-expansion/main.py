from ingest import LegalDocumentIngestor
from retriever import LegalRetriever


def print_documents(title, documents):

    print("\n" + "=" * 70)
    print(title)
    print("=" * 70)

    for index, document in enumerate(documents, start=1):

        print(f"\nResult {index}")

        print("-" * 70)

        print(document[:350])

        print("-" * 70)


def main():

    print("\nBuilding Vector Database...\n")

    ingestor = LegalDocumentIngestor()

    ingestor.ingest()

    retriever = LegalRetriever()

    while True:

        question = input(
            "\nAsk a Legal Question (or type 'exit'): "
        )

        if question.lower() == "exit":
            break

        single_results = retriever.single_query_search(
            question
        )

        print_documents(
            "Single Query Retrieval",
            single_results
        )

        multi_results = retriever.multi_query_search(
            question
        )

        print_documents(
            "Multi Query Retrieval",
            multi_results
        )


if __name__ == "__main__":
    main()