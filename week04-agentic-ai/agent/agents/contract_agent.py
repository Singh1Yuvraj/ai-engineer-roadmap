import time
# pyrefly: ignore [missing-import]
from agent.state import AgentState
# pyrefly: ignore [missing-import]
from typing import Any


class ContractAgent:
    name = "ContractAgent"

    def __init__(self, tool_registry: Any = None):
        self.registry = tool_registry

    def run(self, state: AgentState) -> AgentState:
        start_time = time.time()
        chunks = state.retrieved_documents

        findings = []
        citations = []
        for chunk in chunks:
            citations.append(
                f"{chunk.document_name} (Chunk #{chunk.chunk_id}, Score: {chunk.similarity_score})"
            )
            findings.append(f"Analysis of {chunk.doc_id}: '{chunk.text}'")

        analysis_data = {
            "findings": findings,
            "citations": citations,
            "summary": f"Analyzed {len(chunks)} legal provision(s) across {len(set(c.document_name for c in chunks))} document(s).",
        }

        duration = (time.time() - start_time) * 1000
        state.contract_analysis = analysis_data
        state.add_execution_log(
            agent_name=self.name,
            input_summary=f"Processed {len(chunks)} document chunks",
            output_summary=f"Parsed provisions with citations: {', '.join(citations)}",
            duration_ms=duration,
            confidence=0.91,
        )
        return state