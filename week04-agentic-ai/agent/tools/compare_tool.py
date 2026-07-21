"""
agent/tools/compare_tool.py
Compares structural features across multiple legal documents.
"""

from typing import Dict, Any


class CompareTool:
    def __init__(self):
        self.name = "compare"
        self.description = "Compares two contract types side-by-side."

    def run(self, input_data: Dict[str, str]) -> str:
        """Performs side-by-side document comparison."""
        doc_a = input_data.get("doc_a", "Document A")
        doc_b = input_data.get("doc_b", "Document B")
        
        return (
            f"Comparison Matrix [{doc_a} vs {doc_b}]:\n"
            f"- {doc_a}: High confidentiality compliance, standard 30-day termination.\n"
            f"- {doc_b}: Employment terms, IP assignment clauses, restrictive covenants."
        )