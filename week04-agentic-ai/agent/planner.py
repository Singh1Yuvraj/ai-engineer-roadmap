"""
agent/planner.py
Rule-based planning engine for task decomposition and tool selection.
"""

from typing import List, Dict, Any


class RuleBasedPlanner:
    def __init__(self):
        pass

    def create_plan(self, query: str) -> List[Dict[str, Any]]:
        """
        Analyzes query intent via keyword heuristics and generates a multi-step execution plan.
        """
        q_lower = query.lower()
        plan = []

        # Intent: Contract Comparison
        if "compare" in q_lower or "versus" in q_lower or "vs" in q_lower:
            plan.append({
                "step": 1,
                "tool": "compare",
                "input": {"doc_a": "nda.txt", "doc_b": "employment.txt"}
            })

        # Intent: Clause Extraction
        elif "extract" in q_lower or "clause" in q_lower:
            clause_type = "termination" if "termination" in q_lower else "confidentiality"
            plan.append({
                "step": 1,
                "tool": "extract_clause",
                "input": clause_type
            })

        # Intent: Summarization
        elif "summarize" in q_lower or "summary" in q_lower:
            plan.append({
                "step": 1,
                "tool": "search",
                "input": query
            })
            plan.append({
                "step": 2,
                "tool": "summarize",
                "input": "use_search_results"
            })

        # Default Fallback: Vector / Hybrid Search
        else:
            plan.append({
                "step": 1,
                "tool": "search",
                "input": query
            })

        return plan