"""
embeddings.py
Handles dense vector embedding generation using sentence-transformers.
"""

from typing import List
from sentence_transformers import SentenceTransformer


class EmbeddingEngine:
    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        """
        Initializes the embedding model.
        Default: all-MiniLM-L6-v2 (384 dimensions, fast and accurate).
        """
        print(f"[Embeddings] Loading model: {model_name}...")
        self.model = SentenceTransformer(model_name)
        self.dimension = self.model.get_embedding_dimension()
        print(f"[Embeddings] Model loaded. Embedding dimensions: {self.dimension}")

    def get_embeddings(self, texts: List[str], batch_size: int = 32) -> List[List[float]]:
        """
        Generates dense vector representations for a list of text chunks.
        
        Args:
            texts: List of document/chunk strings.
            batch_size: Number of chunks to process simultaneously.
            
        Returns:
            A list of floating-point vectors.
        """
        if not texts:
            return []
            
        # encode returns numpy arrays; we convert to standard Python floats for database safety
        embeddings = self.model.encode(
            texts, 
            batch_size=batch_size, 
            show_progress_bar=False,
            convert_to_numpy=True
        )
        return embeddings.tolist()

    def get_query_embedding(self, query: str) -> List[float]:
        """
        Generates a dense vector for a single search string.
        """
        return self.model.encode(query, convert_to_numpy=True).tolist()


if __name__ == "__main__":
    # Quick module isolation test
    print("[Testing EmbeddingEngine Isolation]")
    engine = EmbeddingEngine()
    test_vecs = engine.get_embeddings(["Hello legal contract context.", "Another test chunk."])
    print(f"Generated {len(test_vecs)} vectors. Vector length: {len(test_vecs[0])}")