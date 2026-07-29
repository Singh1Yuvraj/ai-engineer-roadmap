import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


@dataclass
class DocumentChunk:
    doc_id: str
    document_name: str
    chunk_id: int
    similarity_score: float
    text: str


@dataclass
class ExecutionRecord:
    agent_name: str
    input_summary: str
    output_summary: str
    duration_ms: float
    confidence: float


class AgentState:

    def __init__(self, user_query: str):
        self.user_query: str = user_query
        self.retrieved_documents: List[DocumentChunk] = []
        self.contract_analysis: Dict[str, Any] = {}
        self.risk_analysis: Dict[str, Any] = {}
        self.execution_records: List[ExecutionRecord] = []
        self.agent_scratchpad: List[str] = []
        self.final_answer: str = ""
        self.total_pipeline_time_ms: float = 0.0

    def add_execution_log(
        self,
        agent_name: str,
        input_summary: str,
        output_summary: str,
        duration_ms: float,
        confidence: float,
    ) -> None:
        record = ExecutionRecord(
            agent_name=agent_name,
            input_summary=input_summary,
            output_summary=output_summary,
            duration_ms=duration_ms,
            confidence=confidence,
        )
        self.execution_records.append(record)
        log_msg = f"[{agent_name}] ({duration_ms:.1f}ms | Conf: {confidence*100:.0f}%): {output_summary}"
        self.agent_scratchpad.append(log_msg)