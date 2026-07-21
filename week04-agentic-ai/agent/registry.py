"""
agent/registry.py
Central registry mapping tool names to execution instances.
"""

from typing import Dict, Any
# pyrefly: ignore [missing-import]
from agent.tools.search_tool import LegalSearchTool
# pyrefly: ignore [missing-import]
from agent.tools.summarize_tool import SummarizeTool
# pyrefly: ignore [missing-import]
from agent.tools.compare_tool import CompareTool
# pyrefly: ignore [missing-import]
from agent.tools.extract_clause_tool import ExtractClauseTool


class ToolRegistry:
    def __init__(self):
        self._tools: Dict[str, Any] = {}
        self._register_default_tools()

    def _register_default_tools(self):
        self.register(LegalSearchTool())
        self.register(SummarizeTool())
        self.register(CompareTool())
        self.register(ExtractClauseTool())

    def register(self, tool_instance: Any):
        self._tools[tool_instance.name] = tool_instance

    def get_tool(self, name: str) -> Any:
        tool = self._tools.get(name)
        if not tool:
            raise ValueError(f"Tool '{name}' is not registered.")
        return tool

    def list_tools(self) -> Dict[str, str]:
        return {name: tool.description for name, tool in self._tools.items()}