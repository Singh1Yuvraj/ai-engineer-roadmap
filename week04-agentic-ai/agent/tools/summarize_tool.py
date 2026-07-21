"""
agent/tools/summarize_tool.py
Summarizes retrieved text contexts.
"""

from typing import Any


class SummarizeTool:
    def __init__(self):
        self.name = "summarize"
        self.description = "Condenses and formats retrieved legal contexts."

    def run(self, text_or_data: Any) -> str:
        """Rule-based text summarization and cleanup."""
        if isinstance(text_or_data, list):
            combined = " ".join([item.get("document", "") for item in text_or_data])
            return f"Summary: {combined[:150]}..."
        return f"Summary: {str(text_or_data)[:150]}..."