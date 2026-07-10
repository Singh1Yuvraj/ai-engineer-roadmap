"""
rrf.py
Reciprocal Rank Fusion (RRF) module.
Combines multiple retrieval runs based solely on document rank order.
"""

from typing import List, Dict, Any


class RRFComposer:
    def __init__(self, k: int = 60):
        """
        Initializes the RRF Composer.
        
        Args:
            k: A constant penalty parameter that dampens the influence of 
               low-ranked documents. Default 60 is standard in information retrieval literature.
        """
        self.k = k

    def fuse(self, retrieval_runs: List[List[Dict[str, Any]]], top_k: int = 3) -> List[Dict[str, Any]]:
        """
        Fuses multiple lists of retrieval results into a single ranked list using RRF.
        
        Args:
            retrieval_runs: A list containing lists of results from different retrievers
                             (e.g., [bm25_results, dense_results]). Each item in a run 
                             must contain an "id", "document", and "metadata".
            top_k: Number of final fused documents to return.
            
        Returns:
            A sorted list of combined results with computed RRF scores.
        """
        rrf_registry: Dict[str, Dict[str, Any]] = {}

        # Iterate over each independent retrieval run (e.g., Run 1 = Lexical, Run 2 = Dense)
        for run in retrieval_runs:
            for rank_idx, hit in enumerate(run):
                doc_id = hit["id"]
                # Rank starts at 1, not 0
                rank = rank_idx + 1
                
                # Calculate reciprocal rank score piece
                reciprocal_rank_score = 1.0 / (self.k + rank)

                if doc_id not in rrf_registry:
                    rrf_registry[doc_id] = {
                        "id": doc_id,
                        "document": hit["document"],
                        "metadata": hit["metadata"],
                        "score": reciprocal_rank_score
                    }
                else:
                    rrf_registry[doc_id]["score"] += reciprocal_rank_score

        # Sort combined map items descending by computed RRF score
        fused_results = list(rrf_registry.values())
        fused_results = sorted(fused_results, key=lambda x: x["score"], reverse=True)
        
        return fused_results[:top_k]

    def print_results(self, results: List[Dict[str, Any]], title: str = "RRF Fused Search"):
        """
        Prints detailed RRF metrics.
        """
        print(f"\n======== {title} (Total Output: {len(results)}) ========")
        for i, res in enumerate(results):
            print(f"Rank {i+1} | Combined RRF Score: {res['score']:.6f}")
            print(f"ID: {res['id']}")
            print(f"Snippet: {res['document'][:90]}...")
            print("-" * 55)


if __name__ == "__main__":
    print("[Testing RRFComposer System Isolation]")
    
    # Mock output matching the structural schema generated from bm25.py and vector_store.py
    mock_bm25_run = [
        {"id": "doc_hr", "document": "Termination policies require 30 days notice.", "metadata": {}},
        {"id": "doc_nda", "document": "NDA provisions bind both parties.", "metadata": {}}
    ]
    
    mock_dense_run = [
        {"id": "doc_nda", "document": "NDA provisions bind both parties.", "metadata": {}},
        {"id": "doc_misc", "document": "Office blueprint information and floor map.", "metadata": {}}
    ]

    composer = RRFComposer(k=60)
    fused_hits = composer.fuse([mock_bm25_run, mock_dense_run], top_k=3)
    composer.print_results(fused_hits)