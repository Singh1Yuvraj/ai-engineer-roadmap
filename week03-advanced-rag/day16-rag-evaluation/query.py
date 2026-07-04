import chromadb

from sentence_transformers import SentenceTransformer

model = SentenceTransformer(
    "all-MiniLM-L6-v2"
)

client = chromadb.PersistentClient(
    path="chroma_db"
)

collection = client.get_collection(
    "legal_documents"
)


class LegalRetriever:

    def retrieve(self, query, k=3):

        query_embedding = model.encode(
            query
        ).tolist()

        results = collection.query(
            query_embeddings=[query_embedding],
            n_results=k
        )

        return {
            "documents": results["documents"][0],
            "metadatas": results["metadatas"][0]
        }