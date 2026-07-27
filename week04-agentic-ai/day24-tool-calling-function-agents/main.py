import os
import sys

# Ensure root directory is in python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from agent.agent import FunctionAgent
from agent.registry import ToolRegistry
from agent.tools.compare_tool import CompareTool
from agent.tools.extract_clause_tool import ExtractClauseTool
from agent.tools.search_tool import SearchTool
from agent.tools.summarize_tool import SummarizeTool


class UpdatedMockLLM:

    def invoke(self, prompt: str) -> str:
        # 1. PLANNER PROMPTS
        if "Available Tool Schemas:" in prompt:
            query = prompt.split("User Query:")[-1].split("\nReturn")[0].lower()

            if "summarize" in query and "termination" in query:
                return """[
                  {"tool": "extract_clause", "arguments": {"clause_type": "termination"}},
                  {"tool": "summarize", "arguments": {"content": "$last"}}
                ]"""
            elif "confidentiality" in query:
                return '[{"tool": "extract_clause", "arguments": {"clause_type": "confidentiality"}}]'
            elif "termination" in query:
                return '[{"tool": "extract_clause", "arguments": {"clause_type": "termination"}}]'
            elif "summarize" in query:
                return '[{"tool": "summarize", "arguments": {"content": "NDA document content"}}]'
            elif "compare" in query:
                return '[{"tool": "compare", "arguments": {"doc_a": "nda.txt", "doc_b": "employment.txt"}}]'
            else:
                return (
                    '[{"tool": "search", "arguments": {"query": "'
                    + query.strip()
                    + '"}}]'
                )

        # 2. SYNTHESIS PROMPTS
        elif "Synthesize a concise" in prompt or "AgentState" in prompt:
            if "CONFIDENTIALITY" in prompt:
                return "The confidentiality clause requires both parties to maintain strict confidentiality of proprietary data for 2 years."
            elif "TERMINATION" in prompt and "SUMMARY" in prompt:
                return "The termination clause was extracted and summarized: either party may terminate upon 30 days written notice."
            elif "TERMINATION" in prompt:
                return "The termination clause states that either party may terminate upon 30 days written notice."
            elif "SUMMARY" in prompt:
                return "The NDA is a standard 2-year agreement protecting proprietary information."
            else:
                return "The requested information was successfully retrieved and processed."

        return "[]"


def main():
    registry = ToolRegistry()
    registry.register(SearchTool())
    registry.register(ExtractClauseTool())
    registry.register(CompareTool())
    registry.register(SummarizeTool())

    llm = UpdatedMockLLM()
    agent = FunctionAgent(llm_client=llm, registry=registry)

    print("\n--- TEST 1: CONFIDENTIALITY QUERY ---")
    ans1 = agent.run("Find confidentiality obligations.")
    print(f"\n🎯 [Final Response]:\n{ans1}\n")

    print("\n--- TEST 2: MULTI-STEP OBSERVATION CHAINING ($last) ---")
    ans2 = agent.run("Summarize termination clause from NDA.")
    print(f"\n🎯 [Final Response]:\n{ans2}\n")


if __name__ == "__main__":
    main()