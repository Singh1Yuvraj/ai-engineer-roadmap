"""
hybrid.py
Hybrid Retrieval engine uniting lexical (BM25) and dense vector search
using Normalized Linear Score Combination.
"""

from PIL import ImageOps
from PIL import ImageOps
from PIL import ImageOps
import os
from typing import List, Dict, Any, Optional
from bm25 import BM25Retriever
from embeddings import EmbeddingEngine
from vector_store import ChromaVectorStore


class HybridRetriever:
    def __init__(
        self, 
        bm25_retriever: BM25Retriever, 
        embedding_engine: EmbeddingEngine, 
        vector_store: ChromaVectorStore
    ):
        """
        Initializes the hybrid retriever by binding the underlying components.
        """
        self.bm25 = bm25_retriever
        self.encoder = embedding_engine
        self.vector_store = vector_store

    def retrieve(
        self, 
        query: str, 
        top_k: int = 3, 
        dense_weight: float = 0.5,
        alpha_lexical: Optional[float] = None
    ) -> List[Dict[str, Any]]:
        """
        Executes a hybrid search across both lexical and dense indices,
        merging the results via a weighted linear combination of normalized scores.
        
        Args:
            query: The text search string.
            top_k: Total number of final sorted results to return.
            dense_weight: The weight assigned to dense retrieval (0.0 to 1.0).
                         Lexical weight will automatically be (1.0 - dense_weight).
            alpha_lexical: Alternative direct assignment parameter. If provided,
                           it overrides dense_weight.
                           
        Returns:
            A combined, re-ranked list of hits sorted by their hybrid score.
        """
        # Determine internal weights
        w_dense = dense_weight
        if alpha_lexical is not None:
            w_lexical = alpha_lexical
            w_dense = 1.0 - w_lexical
        else:
            w_lexical = 1.0 - w_dense

        # 1. Fetch Lexical Matches (Requesting normalized scores)
        # We query a slightly larger depth (top_k * 2) to ensure a healthy overlap pool
        bm25_hits = self.bm25.retrieve(query, top_k=top_k * 2, filter_zeros=True, normalize=True)
        
        # 2. Fetch Dense Matches
        query_vector = self.encoder.get_query_embedding(query)
        dense_hits = self.vector_store.retrieve(query_vector, top_k=top_k * 2, normalize=True)

        # Dictionary to track unified hybrid documents by their ID
        # Structure: id -> { "id", "document", "metadata", "bm25_score", "dense_score" }
        hybrid_registry: Dict[str, Dict[str, Any]] = {}

        # Process Lexical matches
        for hit in bm25_hits:
            doc_id = hit["id"]
            hybrid_registry[doc_id] = {
                "id": doc_id,
                "document": hit["document"],
                "metadata": hit["metadata"],
                "bm25_score": hit["score"],
                "dense_score": 0.0  # Default value if not found in dense path
            }

        # Process Dense matches
        for hit in dense_hits:
            doc_id = hit["id"]
            if doc_id in hybrid_registry:
                hybrid_registry[doc_id]["dense_score"] = hit["score"]
            else:
                hybrid_registry[doc_id] = {
                    "id": doc_id,
                    "document": hit["document"],
                    "metadata": hit["metadata"],
                    "bm25_score": 0.0,  # Default value if not found in lexical path
                    "dense_score": hit["score"]
                }

        # 3. Calculate Final Hybrid Scores
        final_results = []
        for doc_id, info in hybrid_registry.items():
            # Linear score formula application
            hybrid_score = (w_lexical * info["bm25_score"]) + (w_dense * info["dense_score"])
            
            final_results.append({
                "id": doc_id,
                "document": info["document"],
                "score": float(hybrid_score),
                "metadata": info["metadata"],
                "_details": {
                    "normalized_bm25": info["bm25_score"],
                    "normalized_dense": info["dense_score"]
                }
            })

        # Sort combined documents descending by calculated hybrid score
        final_results = sorted(final_results, key=lambda x: x["score"], reverse=True)
        return final_results[:top_k]

    def print_results(self, results: List[Dict[str, Any]], title: str = "Hybrid Retrieval"):
        """
        Prints detailed hybrid diagnostics cleanly.
        """
        print(f"\n======== {title} (Total Output: {len(results)}) ========")
        for i, res in enumerate(results):
            det = res["_details"]
            print(f"Rank {i+1} | Combined Score: {res['score']:.4f}")
            print(f"ID: {res['id']} [Lexical part: {det['normalized_bm25']:.2f}, Dense part: {det['normalized_dense']:.2f}]")
            print(f"Snippet: {res['document'][:90]}...")
            print(f"Metadata: {res['metadata']}")
            print("-" * 65)


if __name__ == "__main__":
    import shutil
    print("[Testing HybridRetriever System Isolation]")
    
    # Setup fresh, local mock resources
    test_db = "hybrid_test_chroma"
    if os.path.exists(test_db):
        shutil.rmtree(test_db)

    # Initialize modules
    lexical_engine = BM25Retriever()
    embedding_engine = EmbeddingEngine()
    dense_store = ChromaVectorStore(persist_directory=test_db)
    
    # Production dummy mock content
    corpus = [
        "Confidentiality Policy: All project blueprints must be secured on company servers.",
        "Termination terms: Either party may end employment with 30 days written notification.",
        "General office layout information regarding building cafeterias and parking structures."
    ]
    uids = ["doc_nda", "doc_hr", "doc_misc"]
    metas = [{"cat": "legal"}, {"cat": "hr"}, {"cat": "facility"}]

    # Index into both systems
    print("\n--- Indexing Core Content ---")
    lexical_engine.index_documents(corpus, doc_ids=uids, metadatas=metas)
    
    computed_vectors = embedding_engine.get_embeddings(corpus)
    dense_store.add_documents(corpus, computed_vectors, uids, metadatas=metas)

    # Instantiate search conductor
    orchestrator = HybridRetriever(lexical_engine, embedding_engine, dense_store)
    
    # Execute query favoring semantic matching slightly (dense_weight=0.6)
    search_query = "secure data terms for termination"
    combined_hits = orchestrator.retrieve(search_query, top_k=2, dense_weight=0.6)
    
    # Display results
    orchestrator.print_results(combined_hits, title="Linear Combined Diagnostic Test")

    # Clean up environment artifacts
    if os.path.exists(test_db):
        shutil.rmtree(test_db)