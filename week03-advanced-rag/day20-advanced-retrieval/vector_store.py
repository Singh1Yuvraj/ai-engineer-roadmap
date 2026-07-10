"""
vector_store.py
Persistent dense vector storage and retrieval system leveraging ChromaDB.
"""

import os
from typing import List, Dict, Any, Optional
import chromadb
from chromadb.config import Settings


class ChromaVectorStore:
    def __init__(self, persist_directory: str = "chroma_db", collection_name: str = "legal_documents"):
        """
        Initializes persistent local ChromaDB storage client.
        """
        self.persist_directory = persist_directory
        self.collection_name = collection_name
        
        # Initialize the persistent client
        self.client = chromadb.PersistentClient(path=self.persist_directory)
        
        # Get or create the vector collection
        # We handle embeddings manually via embeddings.py, so we override Chroma's default embedding function
        self.collection = self.client.get_or_create_collection(
            name=self.collection_name,
            metadata={"hnsw:space": "cosine"} # Using cosine distance for predictable vector scaling
        )
        print(f"[Vector Store] Connected to ChromaDB collection: '{self.collection_name}'")

    def add_documents(
        self, 
        documents: List[str], 
        embeddings: List[List[float]], 
        doc_ids: List[str], 
        metadatas: Optional[List[Dict[str, Any]]] = None
    ):
        """
        Inserts document chunks and their pre-computed dense vectors into the index.
        """
        if not documents or not embeddings or not doc_ids:
            raise ValueError("Documents, embeddings, and doc_ids must not be empty.")
            
        if len(documents) != len(embeddings) != len(doc_ids):
            raise ValueError("All inputs (docs, embeddings, ids) must have matching lengths.")

        if metadatas and len(metadatas) != len(documents):
            raise ValueError("Metadatas length must match documents length.")

        # Chroma expects empty metadata structures if none are passed
        if metadatas is None:
            metadatas = [{} for _ in documents]

        self.collection.add(
            ids=doc_ids,
            embeddings=embeddings,
            documents=documents,
            metadatas=metadatas
        )
        print(f"[Vector Store] Indexed {len(documents)} dense vectors successfully.")

    def retrieve(
        self, 
        query_embedding: List[float], 
        top_k: int = 3,
        normalize: bool = False
    ) -> List[Dict[str, Any]]:
        """
        Queries the vector index using a dense vector embedding.
        
        Returns:
            Structured list of matches mirroring the design structure of bm25.py.
        """
        raw_results = self.collection.query(
            query_embeddings=[query_embedding],
            n_results=top_k,
            include=["documents", "metadatas", "distances"]
        )

        # Chroma structures nested lists since it supports batch queries; extract the first query's data
        ids = raw_results["ids"][0] if raw_results["ids"] else []
        documents = raw_results["documents"][0] if raw_results["documents"] else []
        metadatas = raw_results["metadatas"][0] if raw_results["metadatas"] else []
        distances = raw_results["distances"][0] if raw_results["distances"] else []

        results = []
        for i in range(len(ids)):
            # Convert cosine distance to a similarity score (1.0 - distance)
            # This aligns it with BM25's direction where a higher score means better relevance
            similarity_score = 1.0 - float(distances[i])
            
            results.append({
                "id": ids[i],
                "document": documents[i],
                "score": similarity_score,
                "metadata": metadatas[i] or {}
            })

        # Min-Max Normalization to ensure vector scores fall cleanly between 0.0 and 1.0
        if normalize and results:
            max_score = results[0]["score"]
            min_score = results[-1]["score"]
            score_range = max_score - min_score
            
            for res in results:
                if score_range > 0:
                    res["score"] = (res["score"] - min_score) / score_range
                else:
                    res["score"] = 1.0
                    
        return results

    def print_results(self, results: List[Dict[str, Any]], title: str = "Dense Vector Retrieval"):
        """
        Isolated display interface utility.
        """
        print(f"\n=== {title} (Total Returned: {len(results)}) ===")
        for i, res in enumerate(results):
            print(f"Rank {i+1} | ID: {res['id']} | Similarity Score: {res['score']:.4f}")
            print(f"Snippet: {res['document'][:90]}...")
            print(f"Metadata: {res['metadata']}")
            print("-" * 50)


if __name__ == "__main__":
    import shutil
    print("[Testing ChromaVectorStore Isolation]")
    
    # Simple temporary dir execution check
    test_dir = "test_chroma_db"
    if os.path.exists(test_dir):
        shutil.rmtree(test_dir)
        
    store = ChromaVectorStore(persist_directory=test_dir)
    
    # Mock text and arbitrary 3-dimension dummy embedding vector representations
    mock_docs = ["NDA Policy Agreement", "Employment Terms"]
    mock_vecs = [[0.1, 0.9, 0.2], [0.8, 0.1, 0.4]]
    mock_ids = ["id_1", "id_2"]
    
    store.add_documents(mock_docs, mock_vecs, mock_ids)
    
    # Query with a dummy vector similar to id_1
    hits = store.retrieve(query_embedding=[0.12, 0.88, 0.21], top_k=2, normalize=True)
    store.print_results(hits)
    
    # Clean up test artifact directory
    if os.path.exists(test_dir):
        shutil.rmtree(test_dir)