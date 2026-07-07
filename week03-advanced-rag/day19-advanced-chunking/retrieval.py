import logging
from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional

from embeddings import EmbeddingModel
from vector_store import VectorStore

# Library code should only instantiate the logger, never configure it.
logger = logging.getLogger(__name__)

@dataclass
class SearchResult:
    """Represents a single retrieved chunk with its converted similarity score."""
    id: str
    text: str
    score: float
    metadata: Dict[str, Any] = field(default_factory=dict)


class LegalRetriever:
    """
    Coordinates between the EmbeddingModel and VectorStore to retrieve relevant legal text.
    Handles distance-to-similarity conversion, validation, and sorting.
    """
    
    def __init__(self, embedding_model: EmbeddingModel, vector_store: VectorStore):
        self.embedding_model = embedding_model
        self.vector_store = vector_store

    def retrieve(
        self, 
        query: str, 
        top_k: int = 5, 
        metadata_filter: Optional[Dict[str, Any]] = None
    ) -> List[SearchResult]:
        """
        Embeds the query, retrieves raw distances from the vector store, 
        converts them to similarity scores, and returns sorted results.
        """
        if not query.strip():
            raise ValueError("Query cannot be empty or whitespace.")

        logger.debug(f"Retrieving top {top_k} results for query: '{query}'")
        
        try:
            # 1. Embed the user query
            query_embedding = self.embedding_model.embed_query(query)
            
            # 2. Search the vector database
            raw_results = self.vector_store.similarity_search(
                query_embedding=query_embedding,
                top_k=top_k,
                metadata_filter=metadata_filter
            )
            
            # 3. Convert raw VectorStore outputs (distances) to SearchResult objects (similarity)
            search_results = []
            for res in raw_results:
                # ChromaDB returns distance (closer to 0 is better). 
                # We convert to similarity (closer to 1 is better) for standard thresholding.
                distance = res.get("score", 1.0)
                similarity = 1.0 - distance
                
                search_results.append(
                    SearchResult(
                        id=res.get("id", ""),
                        text=res.get("text", ""),
                        score=similarity,
                        metadata=res.get("metadata", {})
                    )
                )
                
            # 4. Guarantee results are sorted by highest similarity score first
            search_results.sort(key=lambda x: x.score, reverse=True)
            
            return search_results

        except Exception as e:
            logger.error(f"Error during retrieval for query '{query}': {e}")
            raise

    def retrieve_with_threshold(
        self, 
        query: str, 
        threshold: float = 0.75, 
        top_k: int = 5,
        metadata_filter: Optional[Dict[str, Any]] = None
    ) -> List[SearchResult]:
        """
        Retrieves results and filters out matches below the desired similarity threshold.
        """
        results = self.retrieve(query=query, top_k=top_k, metadata_filter=metadata_filter)
        
        # Filter based on the newly converted similarity score
        filtered_results = [res for res in results if res.score >= threshold]
        
        logger.info(f"Retained {len(filtered_results)} out of {len(results)} results above similarity threshold {threshold}.")
        return filtered_results

    def format_results(self, query: str, results: List[SearchResult]) -> str:
        """
        Formats the retrieved results into a highly readable string.
        (Does not print to stdout directly, maintaining separation of concerns).
        """
        lines = []
        lines.append("=" * 50)
        lines.append("Query\n")
        lines.append(query)
        lines.append("=" * 50)
        
        for index, result in enumerate(results, start=1):
            source = result.metadata.get("source", "Unknown")
            chunk_id = result.metadata.get("chunk_id", "Unknown")
            
            lines.append(f"Rank {index}")
            lines.append("\nScore")
            lines.append(f"{result.score:.2f}")
            lines.append("\nSource")
            lines.append(str(source))
            lines.append("\nChunk")
            lines.append(str(chunk_id))
            lines.append("-" * 34)
            lines.append(result.text.strip())
            lines.append("=" * 50)
            
        return "\n".join(lines)