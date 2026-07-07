"""
embeddings.py

A high-performance, production-grade text embedding module.
Handles dense vector extraction while enforcing single-responsibility boundaries.
"""

import logging
import threading
from typing import List, Optional, Union
import numpy as np
from sentence_transformers import SentenceTransformer

# Configure module-level logging
logger = logging.getLogger("embeddings")
if not logger.handlers:
    handler = logging.StreamHandler()
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    logger.setLevel(logging.INFO)


class EmbeddingModel:
    """
    A Thread-Safe Singleton Wrapper around SentenceTransformer.
    Optimized for raw performance by returning direct NumPy ndarrays instead of Python lists.
    """
    _instance: Optional['EmbeddingModel'] = None
    _lock: threading.Lock = threading.Lock()
    
    # Class-level Constants
    DEFAULT_MODEL: str = "all-MiniLM-L6-v2"
    DEFAULT_BATCH_SIZE: int = 32

    def __new__(cls, *args, **kwargs):
        """Enforces a strict singleton pattern across pipeline components."""
        if not cls._instance:
            with cls._lock:
                if not cls._instance:
                    cls._instance = super(EmbeddingModel, cls).__new__(cls)
        return cls._instance

    def __init__(
        self, 
        model_name: str = DEFAULT_MODEL, 
        device: Optional[str] = None,
        normalize_embeddings: bool = True
    ):
        """
        Initializes the embedding engine execution context.
        Ensures idempotent initialization even if called repeatedly via the singleton constructor.
        """
        # Guard clause ensures initialization logic only executes once
        if hasattr(self, "_initialized") and self._initialized:
            return
            
        self.model_name = model_name
        self.normalize_embeddings = normalize_embeddings
        
        logger.info(f"Initializing core EmbeddingModel framework using backbone: '{self.model_name}'")
        
        try:
            # Device Selection: prioritize user override -> default optimal hardware mapping
            self.model = SentenceTransformer(model_name, device=device)
            self._initialized = True
            logger.info(f"Model successfully anchored onto target hardware device: {self.model.device}")
        except Exception as e:
            logger.error(f"Critical failure initializing backbone transformer '{model_name}': {str(e)}")
            raise e

    def __repr__(self) -> str:
        return (f"EmbeddingModel(model_name='{self.model_name}', "
                f"device='{self.model.device}', "
                f"dimension={self.get_dimension()}, "
                f"normalize_embeddings={self.normalize_embeddings})")

    def embed_documents(self, texts: List[str], batch_size: int = DEFAULT_BATCH_SIZE) -> np.ndarray:
        """
        Transforms clean text chunks into a matrix of dense embeddings.

        Args:
            texts: List of document strings to vector map.
            batch_size: Batch workload segmentation size for GPU/CPU optimization loops.

        Returns:
            A 2D NumPy array of shape (num_documents, embedding_dimension).
        """
        # Comprehensive Empty Document Validation
        if not texts:
            logger.warning("Empty or null document batch provided to embed_documents. Returning empty matrix layout.")
            return np.empty((0, self.get_dimension()), dtype=np.float32)

        # Validate entries aren't entirely whitespace null values
        validated_texts = [t if (t and t.strip()) else " " for t in texts]
        
        logger.debug(f"Processing batch conversion array sizing: {len(validated_texts)} records.")
        
        embeddings = self.model.encode(
            validated_texts,
            batch_size=batch_size,
            convert_to_numpy=True,
            normalize_embeddings=self.normalize_embeddings,
            show_progress_bar=False
        )
        return embeddings

    def embed_query(self, query: str) -> np.ndarray:
        """
        Converts a runtime string search query into a single dense 1D vector matrix.
        """
        if not query or not query.strip():
            raise ValueError("Query validation fault: cannot generate embedding transformations on empty string inputs.")
            
        logger.debug("Generating query target footprint matrix vector.")
        embedding = self.model.encode(
            query,
            convert_to_numpy=True,
            normalize_embeddings=self.normalize_embeddings,
            show_progress_bar=False
        )
        return embedding

    def get_dimension(self) -> int:
        """Returns structural dimension sizing for downstream Vector DB schemas (e.g., 384)."""
        return int(self.model.get_sentence_embedding_dimension())

    def cosine_similarity(self, vector_a: np.ndarray, vector_b: np.ndarray) -> float:
        """
        High performance mathematical computation engine for vector comparisons.
        Optimized under the assumption vectors are pre-normalized by this module.
        """
        # If pre-normalized, cosine similarity reduces to a clean vector dot product
        if self.normalize_embeddings:
            return float(np.dot(vector_a, vector_b))
            
        norm_a = np.linalg.norm(vector_a)
        norm_b = np.linalg.norm(vector_b)
        
        if norm_a == 0 or norm_b == 0:
            return 0.0
            
        return float(np.dot(vector_a, vector_b) / (norm_a * norm_b))

    def print_model_info(self) -> None:
        """Prints a highly visible technical diagnostics dashboard for developer logs."""
        dimension = self.get_dimension()
        max_seq_length = getattr(self.model, "max_seq_length", "Unknown")
        device = getattr(self.model, "device", "Unknown")

        print("\n" + "-" * 35)
        print("Embedding Model Configuration Dashboard")
        print("-" * 35)
        print(f"Model Name:           {self.model_name}")
        print(f"Embedding Dimension:  {dimension}")
        print(f"Max Sequence Length:  {max_seq_length}")
        print(f"Device Assignment:    {device}")
        print(f"Enforced L2 Norm:     {self.normalize_embeddings}")
        print("-" * 35 + "\n")


# =====================================================================
# Verification and Validation Loop Execution
# =====================================================================
if __name__ == "__main__":
    print("--- Starting Production Embeddings Validation Run ---\n")

    # 1. Verify Singleton Constraint & Custom Device Override Configuration
    # (Forces execution explicitly onto CPU for localized structural testing validation)
    instance_1 = EmbeddingModel(model_name=EmbeddingModel.DEFAULT_MODEL, device="cpu")
    instance_2 = EmbeddingModel()
    
    print(f"Singleton Verification Check: Are memory references matching? -> {instance_1 is instance_2}")
    
    # 2. Test __repr__ and info output metrics
    print(f"String Representation Protocol Test: {instance_1}")
    instance_1.print_model_info()

    # 3. Test Bulk Document Matrix Generation Loop (Returning Numpy Arrays)
    chunks = [
        "This is chunk one.",
        "This is chunk two.",
        ""  # Validates system structural behavior against anomalous empty strings
    ]
    
    doc_matrix = instance_1.embed_documents(chunks, batch_size=2)
    print(f"Output Structure Class Archetype: {type(doc_matrix)}")
    print(f"Generated Vector Geometry Dimension Mapping shape layout: {doc_matrix.shape}")
    
    # 4. Single Vector Query Testing Block
    query_vector = instance_1.embed_query("This is query footprint mapping data.")
    print(f"Query Vector Sizing Target Signature: {query_vector.shape}")
    
    # 5. Measure structural cosine parsing behavior performance metrics
    sim_score = instance_1.cosine_similarity(doc_matrix[0], query_vector)
    print(f"Computed Vector Distance Similarity Metric: {sim_score:.4f}")
    
    print("\n--- Pipeline Complete. Architecture Optimization Verified. ---")