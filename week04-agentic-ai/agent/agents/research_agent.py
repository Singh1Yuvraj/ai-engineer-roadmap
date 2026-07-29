import time
# pyrefly: ignore [missing-import]
from agent.state import AgentState, DocumentChunk 
# pyrefly: ignore [missing-import]
from typing import Any


class ResearchAgent:
    name = "ResearchAgent"

    def __init__(self, tool_registry: Any = None):
        self.registry = tool_registry

    def run(self, state: AgentState) -> AgentState:
        start_time = time.time()
        query_lower = state.user_query.lower()

        # Generate rich structured document chunk objects with metadata
        chunks = []
        if "confidentiality" in query_lower:
            chunks.append(
                DocumentChunk(
                    doc_id="chunk_014",
                    document_name="nda.txt",
                    chunk_id=14,
                    similarity_score=0.94,
                    text="The receiving party shall keep all proprietary information, trade secrets, and technical data strictly confidential for a period of two (2) years following disclosure.",
                )
            )
        elif "termination" in query_lower:
            chunks.append(
                DocumentChunk(
                    doc_id="chunk_008",
                    document_name="contract_termination.txt",
                    chunk_id=8,
                    similarity_score=0.91,
                    text="Either party may terminate this agreement without cause upon thirty (30) calendar days written notice delivered to the registered office.",
                )
            )
        else:
            chunks.extend(
                [
                    DocumentChunk(
                        doc_id="chunk_002",
                        document_name="nda.txt",
                        chunk_id=2,
                        similarity_score=0.89,
                        text="NDA Section 4.1: Confidential info includes business plans, software code, and financial metrics.",
                    ),
                    DocumentChunk(
                        doc_id="chunk_011",
                        document_name="employment.txt",
                        chunk_id=11,
                        similarity_score=0.87,
                        text="Employment Section 9: Employee non-compete applies for 12 months post-termination within a 50-mile radius.",
                    ),
                ]
            )

        duration = (time.time() - start_time) * 1000
        state.retrieved_documents = chunks
        state.add_execution_log(
            agent_name=self.name,
            input_summary=f"Query: '{state.user_query}'",
            output_summary=f"Retrieved {len(chunks)} chunk(s) with mean score {sum(c.similarity_score for c in chunks)/len(chunks):.2f}",
            duration_ms=duration,
            confidence=0.93,
        )
        return state