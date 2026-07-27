import os
import sys

# Ensure root directory is in python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from agent.agent import ReasoningAgent
from agent.registry import ToolRegistry
from agent.tools.compare_tool import CompareTool
from agent.tools.extract_clause_tool import ExtractClauseTool
from agent.tools.search_tool import SearchTool
from agent.tools.summarize_tool import SummarizeTool


class ReasoningMockLLM:

    def __init__(self):
        self.call_counter = 0

    def invoke(self, prompt: str) -> str:
        # A. PLANNER PROMPTS
        if "Working Memory / Scratchpad:" in prompt:
            query = (
                prompt.split("User Query:")[-1]
                .split("\nWorking Memory")[0]
                .lower()
            )
            scratchpad = prompt.split("Working Memory / Scratchpad:")[-1]

            # Query 1: Summarize NDA (Single-step)
            if "summarize the nda." in query or "summarize nda" in query:
                if "Observation [summarize]:" in scratchpad:
                    return '{"thought": "I have extracted the NDA summary. I can answer now.", "action": "FINAL_ANSWER"}'
                # FIX: Match SummarizeTool's required parameter 'content'
                return '{"thought": "I need to summarize the NDA document.", "action": "TOOL_CALL", "tool": "summarize", "arguments": {"content": "NDA document terms and non-disclosure obligations"}}'

            # Query 2: Conversation Memory (Follow-up: "Compare it with Employment Agreement")
            elif "compare it with" in query or "employment agreement" in query:
                if "Observation [compare]:" in scratchpad:
                    return '{"thought": "I have completed the comparison based on prior NDA context.", "action": "FINAL_ANSWER"}'
                return '{"thought": "The user refers to \'it\' (NDA from prior turn). I will compare nda.txt with Employment Contract.", "action": "TOOL_CALL", "tool": "compare", "arguments": {"doc_a": "nda.txt", "doc_b": "Employment Contract"}}'

            # Query 3: Multi-step compare & recommend
            elif "recommend the stronger agreement" in query or "confidentiality clauses" in query:
                if "Observation [extract_clause]:" not in scratchpad:
                    return '{"thought": "First, I must extract the confidentiality clause from the agreement.", "action": "TOOL_CALL", "tool": "extract_clause", "arguments": {"clause_type": "confidentiality"}}'
                elif "Observation [compare]:" not in scratchpad:
                    return '{"thought": "Now I will compare the confidentiality clause against the Employment Contract.", "action": "TOOL_CALL", "tool": "compare", "arguments": {"doc_a": "$last", "doc_b": "Employment Contract"}}'
                else:
                    return '{"thought": "I have both the extracted clause and the comparison matrix. I am ready to recommend.", "action": "FINAL_ANSWER"}'

            # Query 4: Iterative reasoning (Find, summarize, identify risks)
            elif "identify legal risks" in query or "termination clause" in query:
                if "Observation [extract_clause]:" not in scratchpad:
                    return '{"thought": "First step: extract the termination clause.", "action": "TOOL_CALL", "tool": "extract_clause", "arguments": {"clause_type": "termination"}}'
                elif "Observation [summarize]:" not in scratchpad:
                    return '{"thought": "Next: summarize the extracted termination clause.", "action": "TOOL_CALL", "tool": "summarize", "arguments": {"content": "$last"}}'
                else:
                    return '{"thought": "I have extracted and summarized the clause. Now I can analyze legal risks and finish.", "action": "FINAL_ANSWER"}'

            else:
                return '{"thought": "Executing default search query.", "action": "TOOL_CALL", "tool": "search", "arguments": {"query": "general"}}'

        # B. SYNTHESIS PROMPTS (Fix routing precedence)
        elif "Synthesize a complete and accurate answer" in prompt:
            # Check for Query 4 (Termination & Risks) first to avoid trigger overlap
            if "termination" in prompt and "SUMMARY" in prompt:
                return "Termination Clause Analysis: Extracted 30-day notice requirement. Identified Risk: 30 days may cause operational disruption during transitions."
            elif "confidentiality" in prompt or "recommend" in prompt:
                return "Recommendation: The NDA offers stronger protection due to strict 2-year confidentiality terms versus standard employment clauses."
            elif "Employment Contract" in prompt or "nda.txt" in prompt:
                return "Comparison result: The NDA (from previous context) requires 30 days notice whereas the Employment Contract requires 14 days."
            elif "NDA document terms" in prompt or "SUMMARY" in prompt:
                return "The NDA is a standard agreement protecting proprietary information for 2 years."
            else:
                return "The requested information has been analyzed and processed successfully."

        return "{}"


def main():
    registry = ToolRegistry()
    registry.register(SearchTool())
    registry.register(ExtractClauseTool())
    registry.register(CompareTool())
    registry.register(SummarizeTool())

    llm = ReasoningMockLLM()
    agent = ReasoningAgent(llm_client=llm, registry=registry)

    # Progressive Test Queries
    queries = [
        "Summarize the NDA.",
        "Compare it with Employment Agreement.",  # Tests conversation memory ('it' = NDA)
        "Compare confidentiality clauses and recommend the stronger agreement.",  # Multi-step reasoning
        "Find the termination clause, summarize it, and identify legal risks.",  # Iterative reflection loop
    ]

    for q in queries:
        answer = agent.run(q)
        print(f"\n🎯 [Final Response]:\n{answer}\n")


if __name__ == "__main__":
    main()