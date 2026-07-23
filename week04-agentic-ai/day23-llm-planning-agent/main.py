import os
import sys

# Ensure root path is accessible
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from agent.agent import LLMPlanningAgent


# Simulated Day 22 Tool Classes
class ExtractClauseTool:

    def run(self, clause_type: str):
        return f"Found [{clause_type.upper()}] Clause: Either party may terminate this agreement upon 30 days written notice."


class SummarizeTool:

    def run(self, doc_name: str):
        return f"Summary for {doc_name}: Standard non-disclosure agreement protecting confidential proprietary information for 2 years."


class CompareTool:

    def run(self, doc_a: str, doc_b: str):
        return f"Comparison Matrix: {doc_a} (12-month term, 30-day notice) vs {doc_b} (At-will, 14-day notice)."


class SearchTool:

    def run(self, query: str):
        return f"Search Results: Found 3 matches for confidentiality obligations under Section 4.1."


# Accurate Mock LLM Router for demonstration
class AccurateMockLLM:

    def invoke(self, prompt: str) -> str:
        # 1. PLANNER CALLS
        if "User Query:" in prompt and "AVAILABLE TOOLS:" in prompt:
            query = prompt.split("User Query:")[-1].split("\nPlan:")[0].lower()

            if "termination" in query:
                return '[{"tool": "extract_clause", "input": {"clause_type": "termination"}}]'
            elif "summarize" in query or "summary" in query:
                return '[{"tool": "summarize", "input": {"doc_name": "nda.txt"}}]'
            elif "compare notice" in query:
                return '[{"tool": "compare", "input": {"doc_a": "nda.txt", "doc_b": "employment.txt"}}]'
            elif "compare" in query:
                return '[{"tool": "compare", "input": {"doc_a": "nda.txt", "doc_b": "employment.txt"}}]'
            elif "confidentiality" in query:
                return '[{"tool": "extract_clause", "input": {"clause_type": "confidentiality"}}]'
            else:
                return (
                    '[{"tool": "search", "input": {"query": "'
                    + query.strip()
                    + '"}}]'
                )

        # 2. FINAL ANSWER SYNTHESIS CALL
        elif "Synthesize a final response" in prompt or "Tool Observations:" in prompt:
            if "Found [TERMINATION]" in prompt:
                return "The termination clause states that either party may terminate this agreement with 30 days written notice."
            elif "Summary for nda.txt" in prompt:
                return "The NDA is a standard agreement protecting proprietary information for a 2-year period."
            elif "Comparison Matrix:" in prompt:
                return "Comparison result: The NDA has a 12-month term with 30-day notice, whereas the Employment Contract is at-will with 14-day notice."
            elif "confidentiality" in prompt:
                return "The confidentiality obligations are detailed in Section 4.1 across 3 specific matches."
            else:
                return "Based on the tool observations, here is the answer to your request."

        return "[]"


def main():
    # Day 22 Tool Registry Reuse
    tool_registry = {
        "search": SearchTool(),
        "extract_clause": ExtractClauseTool(),
        "compare": CompareTool(),
        "summarize": SummarizeTool(),
    }

    llm = AccurateMockLLM()
    agent = LLMPlanningAgent(llm_client=llm, tool_registry=tool_registry)

    test_queries = [
        "Find termination clause.",
        "Summarize NDA.",
        "Compare NDA and Employment Contract.",
        "Find confidentiality obligations.",
        "Compare notice periods.",
    ]

    for query in test_queries:
        final_answer = agent.run(query)
        print(f"\n🎯 [Final Answer Response]:\n{final_answer}\n")


if __name__ == "__main__":
    main()