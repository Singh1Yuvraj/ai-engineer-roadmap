"""
chunkers.py
Advanced chunking strategies for hierarchical document parsing.
"""

import re
from typing import List, Dict, Any

class LegalHierarchicalChunker:
    def __init__(self, child_size: int = 150, overlap: int = 30):
        self.child_size = child_size
        self.overlap = overlap

    def split_into_paragraphs(self, text: str) -> List[str]:
        """Splits large structural layout items by carriage returns."""
        return [p.strip() for p in text.split("\n\n") if p.strip()]

    def fixed_sliding_window(self, text: str, chunk_size: int, overlap: int) -> List[str]:
        """Creates fine-grained word overlapping sequences."""
        words = text.split()
        if len(words) <= chunk_size:
            return [text]
            
        chunks = []
        stride = chunk_size - overlap
        for i in range(0, len(words), stride):
            chunk_words = words[i:i + chunk_size]
            chunks.append(" ".join(chunk_words))
            if i + chunk_size >= len(words):
                break
        return chunks

    def create_parent_child_packages(
        self, 
        raw_text: str, 
        source_name: str
    ) -> List[Dict[str, Any]]:
        """
        Parses whole documents down into linked structural components.
        Returns a collection list containing children structures holding packed metadata fields.
        """
        paragraphs = self.split_into_paragraphs(raw_text)
        packaged_elements = []

        for p_idx, paragraph in enumerate(paragraphs):
            parent_id = f"{source_name}_p_{p_idx}"
            
            # Slice down into small child items
            child_strings = self.fixed_sliding_window(paragraph, self.child_size, self.overlap)
            neighbor_ids = [f"{parent_id}_c_{c_idx}" for c_idx in range(len(child_strings))]

            for c_idx, child_text in enumerate(child_strings):
                child_id = neighbor_ids[c_idx]
                
                # Sibling neighborhood array minus current matching child
                current_neighbors = [nid for nid in neighbor_ids if nid != child_id]

                packaged_elements.append({
                    "child_id": child_id,
                    "child_text": child_text,
                    "parent_id": parent_id,
                    "parent_text": paragraph,
                    "neighbors": current_neighbors,
                    "metadata": {
                        "source": source_name,
                        "section": f"Paragraph {p_idx + 1}",
                        "contract_type": "NDA" if "disclosure" in raw_text.lower() else "Agreement"
                    }
                })
        return packaged_elements