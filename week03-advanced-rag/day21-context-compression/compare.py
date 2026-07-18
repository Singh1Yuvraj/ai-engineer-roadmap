"""
compare.py
Metric evaluation framework measuring retrieval efficiency and context reduction percentages.
"""

from typing import List, Dict, Any
from token_budget import TokenBudgetManager

class ContextMetricsEvaluator:
    def __init__(self):
        self.budget_tool = TokenBudgetManager()

    def analyze_compression_efficiency(
        self, 
        raw_results: List[Dict[str, Any]], 
        compressed_results: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """
        Calculates token usage metrics between standard expansion runs and compression pipelines.
        """
        raw_tokens = self.budget_tool.context_length(raw_results)
        comp_tokens = self.budget_tool.context_length(compressed_results)
        
        token_reduction = raw_tokens - comp_tokens
        reduction_percentage = (token_reduction / raw_tokens * 100.0) if raw_tokens > 0 else 0.0
        
        # Calculate context compression ratio (Raw:Compressed)
        compression_ratio = (raw_tokens / comp_tokens) if comp_tokens > 0 else 1.0

        return {
            "raw_total_tokens": raw_tokens,
            "compressed_total_tokens": comp_tokens,
            "saved_tokens": token_reduction,
            "reduction_efficiency_pct": round(reduction_percentage, 2),
            "compression_ratio": round(compression_ratio, 2)
        }

    def display_report(self, metrics: Dict[str, Any], title: str = "Context Compression Audit"):
        """Prints a scannable performance table."""
        print(f"\n==================================================")
        print(f"📊 {title}")
        print(f"==================================================")
        print(f" Raw Context Volume       : {metrics['raw_total_tokens']} tokens")
        print(f" Compressed Context Volume: {metrics['compressed_total_tokens']} tokens")
        print(f" Saved Context Space      : {metrics['saved_tokens']} tokens")
        print(f"--------------------------------------------------")
        print(f" 🔥 Context Reduction %   : {metrics['reduction_efficiency_pct']}%")
        print(f" 🎯 Compression Ratio     : {metrics['compression_ratio']}:1")
        print(f"==================================================")