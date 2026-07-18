"""
vector_store.py
Persistent dense vector storage and retrieval system leveraging ChromaDB.
"""

import os
from typing import List, Dict, Any, Optional
import chromadb

class ChromaVectorStore:
    def __init__(self, persist_directory: str = "chroma_db", collection_name: str = "compressed_legal_docs"):
        self.persist_directory = persist_directory
        self.collection_name = collection_name
        self.client = chromadb.PersistentClient(path=self.persist_directory)
        self.collection = self.client.get_or_create_collection(
            name=self.collection_name,
            metadata={"hnsw:space": "cosine"}
        )

    def add_documents(
        self, 
        documents: List[str], 
        embeddings: List[List[float]], 
        doc_ids: List[str], 
        metadatas: Optional[List[Dict[str, Any]]] = None
    ):
        if metadatas is None:
            metadatas = [{} for _ in documents]
        self.collection.add(
            ids=doc_ids,
            embeddings=embeddings,
            documents=documents,
            metadatas=metadatas
        )

    def retrieve(
        self, 
        query_embedding: List[float], 
        top_k: int = 3,
        normalize: bool = False
    ) -> List[Dict[str, Any]]:
        raw_results = self.collection.query(
            query_embeddings=[query_embedding],
            n_results=top_k,
            include=["documents", "metadatas", "distances"]
        )

        ids = raw_results["ids"][0] if raw_results["ids"] else []
        documents = raw_results["documents"][0] if raw_results["documents"] else []
        metadatas = raw_results["metadatas"][0] if raw_results["metadatas"] else []
        distances = raw_results["distances"][0] if raw_results["distances"] else []

        results = []
        for i in range(len(ids)):
            similarity_score = 1.0 - float(distances[i])
            results.append({
                "id": ids[i],
                "document": documents[i],
                "score": similarity_score,
                "metadata": metadatas[i] or {}
            })
        return results