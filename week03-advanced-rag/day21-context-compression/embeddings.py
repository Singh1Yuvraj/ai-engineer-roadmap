"""
embeddings.py
Production-ready embedding engine using sentence-transformers.
"""

from typing import List
from sentence_transformers import SentenceTransformer

class EmbeddingEngine:
    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        print(f"[Embeddings] Loading model: {model_name}...")
        self.model = SentenceTransformer(model_name)
        # Updated method for compatibility
        self.dimension = self.model.get_embedding_dimension()
        print(f"[Embeddings] Model loaded. Dimensions: {self.dimension}")

    def get_embeddings(self, texts: List[str], batch_size: int = 32) -> List[List[float]]:
        if not texts:
            return []
        embeddings = self.model.encode(texts, batch_size=batch_size, show_progress_bar=False)
        return embeddings.tolist()

    def get_query_embedding(self, query: str) -> List[float]:
        return self.model.encode(query).tolist()