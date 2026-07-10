"""
mmr.py
Maximal Marginal Relevance (MMR) diversity re-ranking engine.
"""

from typing import List, Dict, Any
import numpy as np


def cosine_similarity(vec1: List[float], vec2: List[float]) -> float:
    """Computes basic cosine similarity between two standard python lists."""
    v1, v2 = np.array(vec1), np.array(vec2)
    norm1 = np.linalg.norm(v1)
    norm2 = np.linalg.norm(v2)
    if norm1 == 0.0 or norm2 == 0.0:
        return 0.0
    return float(np.dot(v1, v2) / (norm1 * norm2))


class MMRReRanker:
    def __init__(self, lambda_mult: float = 0.5):
        """
        Initializes the MMR Engine.
        
        Args:
            lambda_mult: Diversity parameter between 0.0 and 1.0.
                         1.0 = Pure relevance (standard vector search).
                         0.0 = Pure diversity (forces completely distinct content chunks).
        """
        self.lambda_mult = lambda_mult

    def re_rank(
        self, 
        query_embedding: List[float], 
        candidates: List[Dict[str, Any]], 
        candidate_embeddings: List[List[float]], 
        top_k: int = 3
    ) -> List[Dict[str, Any]]:
        """
        Re-ranks a group of candidate matches to maximize relevance and minimize similarity overlap.
        
        Args:
            query_embedding: Dense vector vector representation of search string.
            candidates: A structured list of document match dictionaries.
            candidate_embeddings: Matched tracking list containing dense lists for candidates.
            top_k: Exact max count of distinct hits to pull back out.
        """
        if not candidates:
            return []
            
        top_k = min(top_k, len(candidates))
        
        # Track by indices
        selected_indices: List[int] = []
        remaining_indices = list(range(len(candidates)))
        
        # 1. Precalculate similarities of all documents to the query vector
        query_similarities = [
            cosine_similarity(query_embedding, doc_vec) 
            for doc_vec in candidate_embeddings
        ]

        # 2. Iteratively build the selection pool
        while len(selected_indices) < top_k:
            best_mmr_score = -float("inf")
            best_idx = -1
            
            for idx in remaining_indices:
                rel_score = query_similarities[idx]
                
                # Compute maximum similarity to any document already selected
                if not selected_indices:
                    max_sim_to_selected = 0.0
                else:
                    max_sim_to_selected = max([
                        cosine_similarity(candidate_embeddings[idx], candidate_embeddings[s_idx])
                        for s_idx in selected_indices
                    ])
                
                # Core MMR equation calculation
                mmr_score = (self.lambda_mult * rel_score) - ((1.0 - self.lambda_mult) * max_sim_to_selected)
                
                if mmr_score > best_mmr_score:
                    best_mmr_score = mmr_score
                    best_idx = idx
            
            # Move the best item from remaining to selected
            remaining_indices.remove(best_idx)
            selected_indices.append(best_idx)

        # Assemble output payload sequence
        reranked_results = []
        for rank_pos, final_idx in enumerate(selected_indices):
            hit = candidates[final_idx].copy()
            hit["_mmr_rank"] = rank_pos + 1
            reranked_results.append(hit)
            
        return reranked_results


if __name__ == "__main__":
    print("[Testing MMRReRanker System Isolation]")
    
    # Mock structured matches
    mock_candidates = [
        {"id": "doc_1", "document": "Confidential NDA text copy section A.", "metadata": {}},
        {"id": "doc_2", "document": "Confidential NDA text copy section B (Highly Redundant).", "metadata": {}},
        {"id": "doc_3", "document": "Completely different HR Termination details.", "metadata": {}}
    ]
    
    # 3-Dimensional Mock Vectors
    q_vec = [0.5, 0.5, 0.0]
    mock_vectors = [
        [0.5, 0.4, 0.1],  # Highly similar to query, highly similar to doc_2
        [0.5, 0.4, 0.15], # Highly similar to query, highly similar to doc_1
        [0.1, 0.1, 0.9]   # Lower query similarity, completely distinct composition
    ]
    
    # Requesting high diversity (lambda=0.3) to penalize doc_2 in favor of doc_3
    reranker = MMRReRanker(lambda_mult=0.3)
    fused_hits = reranker.re_rank(q_vec, mock_candidates, mock_vectors, top_k=3)
    
    for item in fused_hits:
        print(f"MMR Rank {item['_mmr_rank']} | ID: {item['id']} -> {item['document']}")