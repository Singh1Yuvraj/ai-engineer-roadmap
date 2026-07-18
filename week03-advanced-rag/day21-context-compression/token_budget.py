"""
token_budget.py
Dynamic token monitoring and context truncation orchestrator.
"""

from typing import List, Dict, Any


class TokenBudgetManager:
    def __init__(self, fallback_chars_per_token: float = 4.0):
        """
        Initializes budget controls. Uses a standard character heuristic (4 characters ~ 1 token) 
        as a fallback if an explicit model tokenizer is missing.
        """
        self.chars_per_token = fallback_chars_per_token

    def estimate_tokens(self, text: str) -> int:
        """Estimates token footprint scale."""
        return max(1, int(len(text) / self.chars_per_token))

    def context_length(self, documents: List[Dict[str, Any]]) -> int:
        """Sums up the total combined token scale across an entire retrieval list."""
        return sum(self.estimate_tokens(d["document"]) for d in documents)

    def remaining_tokens(self, current_context: List[Dict[str, Any]], max_budget: int) -> int:
        """Returns how many empty token slots are left before hitting bounds."""
        return max(0, max_budget - self.context_length(current_context))

    def chunk_budget(self, total_budget: int, item_count: int) -> int:
        """Calculates equal safe chunk bounds allocation sizes."""
        if item_count <= 0:
            return total_budget
        return int(total_budget / item_count)

    def truncate_context(self, documents: List[Dict[str, Any]], max_token_budget: int) -> List[Dict[str, Any]]:
        """
        Progressively builds a context window item by item, dropping trailing 
        documents as soon as the hard token limit threshold is crossed.
        """
        allocated_pool = []
        running_tokens = 0
        
        for doc in documents:
            doc_tokens = self.estimate_tokens(doc["document"])
            if running_tokens + doc_tokens <= max_token_budget:
                allocated_pool.append(doc)
                running_tokens += doc_tokens
            else:
                # Add the final document partially by truncating its text
                available_tokens = max_token_budget - running_tokens
                if available_tokens > 5: # Only bother if enough character length space remains
                    char_limit = int(available_tokens * self.chars_per_token)
                    truncated_doc = doc.copy()
                    truncated_doc["document"] = doc["document"][:char_limit] + " ... [Truncated]"
                    allocated_pool.append(truncated_doc)
                break
                
        return allocated_pool

    def dynamic_budget(self, base_budget: int, user_query: str, system_prompt: str = "") -> int:
        """Adjusts the safe target budget limit by subtracting the query and prompt overhead."""
        overhead = self.estimate_tokens(user_query) + self.estimate_tokens(system_prompt)
        return max(0, base_budget - overhead)