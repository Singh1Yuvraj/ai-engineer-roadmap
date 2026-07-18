"""
retrieval.py
Retriever orchestration engine tying metadata filtering, expansion, and compression.
"""

from typing import List, Dict, Any, Optional
from metadata_filter import MetadataFilter
from parent_child import ParentChildRetriever
from compressor import LegalContextCompressor
from token_budget import TokenBudgetManager

class AdvancedCompressedRetriever:
    def __init__(
        self, 
        vector_store_instance: Any, 
        embedding_engine_instance: Any,
        parent_child_manager: ParentChildRetriever
    ):
        self.store = vector_store_instance
        self.encoder = embedding_engine_instance
        self.pc_manager = parent_child_manager
        
        # Tools
        self.filter_tool = MetadataFilter()
        self.compressor_tool = LegalContextCompressor()
        self.budget_tool = TokenBudgetManager()

    def search(
        self,
        query: str,
        mode: str = "vanilla",  # Mode layout options: 'vanilla', 'parent_expanded', 'compressed'
        top_k: int = 3,
        metadata_filters: Optional[Dict[str, Any]] = None,
        token_limit: int = 1000
    ) -> List[Dict[str, Any]]:
        """
        Unified dynamic context engine execution loop router.
        """
        # 1. Fetch dense base vectors
        query_vector = self.encoder.get_query_embedding(query)
        
        # Pull a slightly deeper set of items to allow for structural breathing room
        raw_hits = self.store.retrieve(query_vector, top_k=top_k * 2, normalize=False)
        
        if not raw_hits:
            return []

        # 2. Apply metadata filters if present
        if metadata_filters:
            raw_hits = self.filter_tool.combine_filters(raw_hits, metadata_filters)

        # 3. Apply retrieval route configurations
        if mode == "vanilla":
            processed_hits = raw_hits[:top_k]
            
        elif mode == "parent_expanded":
            # Expand children to full parents
            processed_hits = self.pc_manager.expand_context(raw_hits, mode="parent")
            processed_hits = processed_hits[:top_k]
            
        elif mode == "compressed":
            # First expand to get full context windows, then compress
            expanded = self.pc_manager.expand_context(raw_hits, mode="parent")
            
            processed_hits = []
            query_keywords = [w.lower() for w in query.split() if len(w) > 3]
            
            for item in expanded:
                comp_doc = item.copy()
                # Run text through compressor suite pipeline metrics
                text_kernel = self.compressor_tool.keyword_compression(item["document"], query_keywords)
                text_kernel = self.compressor_tool.sentence_compression(text_kernel)
                
                comp_doc["document"] = text_kernel
                processed_hits.append(comp_doc)
                
            processed_hits = processed_hits[:top_k]

        # 4. Enforce structural budget allocation boundaries
        final_bounded_context = self.budget_tool.truncate_context(processed_hits, token_limit)
        return final_bounded_context