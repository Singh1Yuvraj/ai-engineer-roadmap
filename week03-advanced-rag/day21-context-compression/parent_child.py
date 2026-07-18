"""
parent_child.py
Parent-Child relationships index parser and context expander.
"""

from typing import List, Dict, Any, Optional


class ParentChildRetriever:
    def __init__(self):
        # Master mapping databases simulating production key store caches
        # Format: child_id -> parent_dict, child_id -> sibling_list
        self.child_to_parent_map: Dict[str, Dict[str, Any]] = {}
        self.child_to_neighbors_map: Dict[str, List[str]] = {}

    def register_relationship(
        self, 
        child_id: str, 
        parent_id: str, 
        parent_text: str, 
        neighbors: List[str],
        metadata: Optional[Dict[str, Any]] = None
    ):
        """Maps relationships between dense child identifiers and their larger components."""
        self.child_to_parent_map[child_id] = {
            "id": parent_id,
            "document": parent_text,
            "metadata": metadata or {}
        }
        self.child_to_neighbors_map[child_id] = neighbors

    def retrieve_child(self, child_hit: Dict[str, Any]) -> Dict[str, Any]:
        """Returns the raw matched narrow vector chunk (the default fallback match)."""
        return child_hit

    def retrieve_parent(self, child_id: str) -> Optional[Dict[str, Any]]:
        """
        Looks up the narrow child ID and swaps it for the broad Parent document structure.
        Ensures perfect preservation of core context details.
        """
        return self.child_to_parent_map.get(child_id)

    def retrieve_neighbors(self, child_id: str, all_candidates_pool: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Fetches the immediate structural context surrounding a child chunk."""
        neighbor_ids = self.child_to_neighbors_map.get(child_id, [])
        if not neighbor_ids:
            return []
            
        # Re-assemble matches out of the parsing cache pool elements
        return [c for c in all_candidates_pool if c.get("id") in neighbor_ids]

    def merge_sections(self, retrieved_chunks: List[Dict[str, Any]]) -> str:
        """De-duplicates overlapping sections and combines text into a clean narrative block."""
        seen_texts = set()
        merged_blocks = []
        
        for chunk in retrieved_chunks:
            text = chunk["document"].strip()
            if text not in seen_texts:
                seen_texts.add(text)
                merged_blocks.append(text)
                
        return "\n\n--- Structural Section Merge ---\n\n".join(merged_blocks)

    def expand_context(self, child_hits: List[Dict[str, Any]], mode: str = "parent") -> List[Dict[str, Any]]:
        """
        Orchestration route that transforms a list of sharp vector hits 
        into expanded parents or context blocks.
        Supported modes: 'child' (pass-through), 'parent' (expansion)
        """
        if mode == "child":
            return child_hits
            
        expanded_list = []
        seen_parents = set()
        
        for hit in child_hits:
            c_id = hit["id"]
            if mode == "parent":
                parent_doc = self.retrieve_parent(c_id)
                if parent_doc:
                    p_id = parent_doc["id"]
                    if p_id not in seen_parents:
                        seen_parents.add(p_id)
                        expanded_list.append(parent_doc)
                else:
                    # Fallback to child if no parent is registered
                    expanded_list.append(hit)
                    
        return expanded_list