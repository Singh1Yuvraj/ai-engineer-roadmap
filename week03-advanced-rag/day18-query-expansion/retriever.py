import chromadb

from sentence_transformers import SentenceTransformer

from query_expansion import QueryExpander

CHROMA_PATH = "chroma"
COLLECTION_NAME = "legal_documents"


class LegalRetriever:

    def __init__(self):

        self.model = SentenceTransformer(
            "all-MiniLM-L6-v2"
        )

        self.client = chromadb.PersistentClient(
            path=CHROMA_PATH
        )

        self.collection = self.client.get_collection(
            COLLECTION_NAME
        )

        self.expander = QueryExpander()

    def single_query_search(
        self,
        question,
        top_k=3
    ):

        embedding = self.model.encode(
            question
        ).tolist()

        results = self.collection.query(

            query_embeddings=[embedding],

            n_results=top_k
        )

        return results["documents"][0]

    def multi_query_search(
        self,
        question,
        top_k=3
    ):

        expanded_queries = self.expander.expand(
            question
        )

        unique_documents = []

        seen = set()

        for query in expanded_queries:

            embedding = self.model.encode(
                query
            ).tolist()

            results = self.collection.query(

                query_embeddings=[embedding],

                n_results=top_k
            )

            for doc in results["documents"][0]:

                if doc not in seen:

                    unique_documents.append(doc)

                    seen.add(doc)

        return unique_documents

    def print_results(
        self,
        documents
    ):

        print()

        for index, document in enumerate(
            documents,
            start=1
        ):

            print(f"{index}.")
            print(document[:250])
            print("-" * 50)


if __name__ == "__main__":

    retriever = LegalRetriever()

    question = "Can employees reveal company secrets?"

    print("\nSingle Query Retrieval")

    single = retriever.single_query_search(
        question
    )

    retriever.print_results(single)

    print("\nMulti Query Retrieval")

    multi = retriever.multi_query_search(
        question
    )

    retriever.print_results(multi)