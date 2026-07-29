import sys
import os

# Ensure root directory is in python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from agent.registry import ToolRegistry
from agent.tools.search_tool import SearchTool
from agent.tools.extract_clause_tool import ExtractClauseTool
from agent.tools.compare_tool import CompareTool
from agent.tools.summarize_tool import SummarizeTool
from agent.multi_agent import MultiAgentSystem

class MockLLM:
    def invoke(self, prompt: str) -> str:
        return "Synthesized multi-agent collaborative output."

def main():
    registry = ToolRegistry()
    registry.register(SearchTool())
    registry.register(ExtractClauseTool())
    registry.register(CompareTool())
    registry.register(SummarizeTool())

    llm = MockLLM()
    system = MultiAgentSystem(llm_client=llm, tool_registry=registry)

    queries = [
        "Analyze confidentiality obligations in the NDA.",
        "Find termination clauses and identify legal risks.",
        "Compare NDA and Employment Contract and recommend which is legally stronger.",
        "Review Vendor Agreement and generate negotiation recommendations."
    ]

    for q in queries:
        report = system.run(q)
        print(f"\n🎯 [Final Output Report]:\n{report}\n")

if __name__ == "__main__":
    main()