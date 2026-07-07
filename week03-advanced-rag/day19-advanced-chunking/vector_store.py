"""
vector_store.py

A clean, high-performance abstraction layer around Persistent ChromaDB storage.
Responsible solely for database management, index storage, and vector retrieval.
"""

import logging
import hashlib
from typing import List, Dict, Any, Optional, Union
import numpy as np
import chromadb
from chromadb.config import Settings

# Configure module-level logging
logger = logging.getLogger("vector_store")
if not logger.handlers:
    handler = logging.StreamHandler()
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    logger.setLevel(logging.INFO)


class VectorStore:
    """
    Manages a persistent ChromaDB instance. Completely decoupled from embedding models
    and text chunkers, accepting pre-computed matrices directly for optimized performance.
    """

    DEFAULT_BATCH_SIZE: int = 100

    def __init__(
        self, 
        collection_name: str = "legal_documents", 
        persist_directory: str = "./chroma_db"
    ):
        """Initializes the persistent ChromaDB client engine."""
        self._collection_name = collection_name
        self.persist_directory = persist_directory
        
        logger.info(f"Initializing Persistent ChromaDB client at path: '{self.persist_directory}'")
        
        self.client = chromadb.PersistentClient(
            path=self.persist_directory,
            settings=Settings(anonymized_telemetry=False)
        )
        
        self.collection = self.create_collection(self._collection_name)

    @property
    def collection_name(self) -> str:
        """Explicit getter returning active collection configuration name."""
        return self._collection_name

    def create_collection(self, name: str) -> Any:
        """Retrieves or creates a named ChromaDB collection without internal embeddings."""
        try:
            collection = self.client.get_or_create_collection(
                name=name,
                embedding_function=None  # Managed externally by embeddings.py
            )
            logger.info(f"Connected to ChromaDB collection index: '{name}'")
            return collection
        except Exception as e:
            logger.critical(f"Failed to initialize ChromaDB collection reference framework: {str(e)}")
            raise e

    def _generate_deterministic_id(self, text: str, metadata: Optional[Dict[str, Any]] = None) -> str:
        """
        Generates a fast, lightweight, deterministic SHA-256 hash string ID.
        Prevents duplicate text-blocks entries from inflating document clusters.
        """
        hasher = hashlib.sha256(text.encode("utf-8", errors="ignore"))
        if metadata:
            # Anchor uniqueness with structural parameters if present
            source_tag = str(metadata.get("source", ""))
            chunk_tag = str(metadata.get("chunk", ""))
            hasher.update(source_tag.encode("utf-8"))
            hasher.update(chunk_tag.encode("utf-8"))
        return hasher.hexdigest()

    def add_documents(
        self, 
        documents: List[str], 
        embeddings: Union[np.ndarray, List[List[float]]], 
        metadatas: Optional[List[Dict[str, Any]]] = None,
        batch_size: int = DEFAULT_BATCH_SIZE
    ) -> List[str]:
        """
        Ingests vectors, texts, and metadata tracking configurations in safe chunks.
        """
        if not documents:
            logger.warning("Empty execution array passed to add_documents. Index task bypassed.")
            return []

        processed_embeddings = embeddings.tolist() if isinstance(embeddings, np.ndarray) else embeddings

        if len(documents) != len(processed_embeddings):
            raise ValueError("Length mismatch between documents and structural tracking embeddings.")
        if metadatas and len(documents) != len(metadatas):
            raise ValueError("Length mismatch between documents and corresponding metadata inputs.")

        # Generate lightweight deterministic structural keys instead of UUID allocations
        generated_ids = [
            self._generate_deterministic_id(doc, metadatas[i] if metadatas else None) 
            for i, doc in enumerate(documents)
        ]

        total_docs = len(documents)
        logger.info(f"Ingesting {total_docs} records into '{self._collection_name}' (Batch Size: {batch_size})...")

        # Process slice intervals safely to avoid database hardware socket limit overflows
        for i in range(0, total_docs, batch_size):
            end_idx = min(i + batch_size, total_docs)
            self.collection.add(
                ids=generated_ids[i:end_idx],
                embeddings=processed_embeddings[i:end_idx],
                metadatas=metadatas[i:end_idx] if metadatas else None,
                documents=documents[i:end_idx]
            )
        
        logger.info("Batch ingestion execution confirmed successfully.")
        return generated_ids

    def update_documents(
        self,
        ids: List[str],
        documents: Optional[List[str]] = None,
        embeddings: Optional[Union[np.ndarray, List[List[float]]]] = None,
        metadatas: Optional[List[Dict[str, Any]]] = None,
        batch_size: int = DEFAULT_BATCH_SIZE
    ) -> None:
        """ Updates existing matching index entries inside the collection layer. """
        processed_embeddings = embeddings.tolist() if isinstance(embeddings, np.ndarray) else embeddings
        total_items = len(ids)

        logger.info(f"Updating {total_items} tracking rows inside vector store collection framework...")
        for i in range(0, total_items, batch_size):
            end_idx = min(i + batch_size, total_items)
            self.collection.update(
                ids=ids[i:end_idx],
                embeddings=processed_embeddings[i:end_idx] if processed_embeddings else None,
                metadatas=metadatas[i:end_idx] if metadatas else None,
                documents=documents[i:end_idx] if documents else None
            )

    def similarity_search(
        self, 
        query_embedding: Union[np.ndarray, List[float]], 
        top_k: int = 5,
        metadata_filter: Optional[Dict[str, Any]] = None,
        content_filter: Optional[Dict[str, Any]] = None
    ) -> List[Dict[str, Any]]:
        """
        Queries the dataset with support for structural metadata and keyword filter syntax.
        """
        processed_query = query_embedding.tolist() if isinstance(query_embedding, np.ndarray) else query_embedding

        if isinstance(processed_query[0], list):
            processed_query = processed_query[0]

        results = self.collection.query(
            query_embeddings=[processed_query],
            n_results=top_k,
            where=metadata_filter,         # Structured fields filtering (e.g., {"source": "doc.txt"})
            where_document=content_filter   # Term string filtering (e.g., {"$contains": "clause"})
        )

        formatted_results = []
        if not results or not results["documents"]:
            return formatted_results

        for i in range(len(results["ids"][0])):
            formatted_results.append({
                "id": results["ids"][0][i],
                "text": results["documents"][0][i],
                "metadata": results["metadatas"][0][i] if results["metadatas"] else {},
                "score": results["distances"][0][i] if results["distances"] else 0.0
            })

        return formatted_results

    def count(self) -> int:
        """Returns the current document volume inside the targeted cluster collection index."""
        return int(self.collection.count())

    def peek(self, limit: int = 3) -> List[Dict[str, Any]]:
        """
        Returns an inspection block array containing a snapshot of stored collection elements.
        Avoids direct standard out terminal side effects.
        """
        total_items = self.count()
        actual_limit = min(limit, total_items)
        
        if actual_limit == 0:
            return []

        data = self.collection.peek(limit=actual_limit)
        peek_records = []
        
        for i in range(actual_limit):
            peek_records.append({
                "id": data["ids"][i],
                "text": data["documents"][i],
                "metadata": data["metadatas"][i] if data["metadatas"] else {}
            })
            
        return peek_records

    def delete_documents(self, ids: List[str]) -> None:
        """Removes documents from index using matching ID array keys."""
        if not ids:
            return
        logger.warning(f"Purging {len(ids)} target records from collection partition space.")
        self.collection.delete(ids=ids)

    def reset_collection(self) -> None:
        """Wipes and recreates the collection, safely confirming existence first to avoid exceptions."""
        logger.warning(f"Executing reset routine on collection: '{self._collection_name}'")
        try:
            # Query active client collections to verify existence safely first
            existing_collections = [c.name for c in self.client.list_collections()]
            
            if self._collection_name in existing_collections:
                self.client.delete_collection(name=self._collection_name)
                logger.debug(f"Old collection '{self._collection_name}' dropped successfully.")
            else:
                logger.debug(f"Collection '{self._collection_name}' did not exist. Bypassing deletion steps.")

            self.collection = self.create_collection(self._collection_name)
            logger.info("Collection wipe reset completed cleanly.")
        except Exception as e:
            logger.error(f"Error resetting active storage elements: {str(e)}")
            raise e

    def get_collection(self) -> Any:
        """Returns the raw internal underlying Chroma collection instance directly."""
        return self.collection


# =====================================================================
# Verification and Validation Loop Execution
# =====================================================================
if __name__ == "__main__":
    print("--- Starting Production Vector Store Functional Test Run ---\n")

    # 1. Initialize Vector Store
    vector_store = VectorStore(collection_name="demo_legal_store", persist_directory="./chroma_db")
    vector_store.reset_collection()

    # 2. Fabricate pipeline parameters
    mock_chunks = [
        "EMPLOYMENT CLAUSE 1: Employee salary parameters match grade level indexing specifications.",
        "NDA DISCLOSURE: Non-disclosure constraints preserve execution viability for 5 years."
    ]
    
    np.random.seed(42)
    mock_embeddings = np.random.rand(2, 384).astype(np.float32)
    
    mock_metadata = [
        {"source": "employment_agreement.txt", "chunk": 0},
        {"source": "nda_document.txt", "chunk": 5}
    ]

    # 3. Add to Database Store (Validates Batch Splitting and Deterministic ID logic)
    inserted_ids = vector_store.add_documents(
        documents=mock_chunks,
        embeddings=mock_embeddings,
        metadatas=mock_metadata,
        batch_size=1
    )
    print(f"Generated Hash IDs: {inserted_ids}")
    print(f"Total Database Rows: {vector_store.count()}")

    # 4. Test Update API
    updated_chunks = ["EMPLOYMENT CLAUSE 1: Updated salary parameters matching corporate review cycles."]
    vector_store.update_documents(ids=[inserted_ids[0]], documents=updated_chunks)

    # 5. Data Return Peek Verification
    snapshot = vector_store.peek(limit=1)
    print(f"\nPeek Returned Count: {len(snapshot)}")
    print(f"Peek Content Fragment: {snapshot[0]['text']}")

    # 6. Execute Advanced Search Queries with Metadata & Content Filters
    mock_query_vector = np.random.rand(1, 384).astype(np.float32)
    
    print("\nExecuting similarity query with 'source' metadata filter restriction...")
    search_results = vector_store.similarity_search(
        query_embedding=mock_query_vector, 
        top_k=1,
        metadata_filter={"source": "employment_agreement.txt"},
        content_filter={"$contains": "salary"}
    )
    
    if search_results:
        print("[Top Filtered Query Match Output]")
        print(f"Matched Text: {search_results[0]['text']}")
        print(f"Score Metric: {search_results[0]['score']:.4f}")
    else:
        print("No matches returned matching filter parameters.")

    print("\n--- Pipeline Complete. DB Interface Decoupling Verified. ---")