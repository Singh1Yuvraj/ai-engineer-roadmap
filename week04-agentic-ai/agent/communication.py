# pyrefly: ignore [missing-import]
from agent.state import AgentState
# pyrefly: ignore [missing-import]
from typing import Any
# pyrefly: ignore [missing-import]
from agent.registry import ToolRegistry
# pyrefly: ignore [missing-import]
from agent.agents.research_agent import ResearchAgent
# pyrefly: ignore [missing-import]
from agent.agents.contract_agent import ContractAgent
# pyrefly: ignore [missing-import]
from agent.agents.risk_agent import RiskAgent
# pyrefly: ignore [missing-import]
from agent.agents.response_agent import ResponseAgent


class CommunicationBus:

    @staticmethod
    def print_trace(state: AgentState) -> None:
        print("\n" + "=" * 70)
        print("📜 MULTI-AGENT EXECUTION GRAPH & TIMINGS TRACE")
        print("=" * 70)
        for record in state.execution_records:
            print(f"\n🔹 Agent: {record.agent_name}")
            print(f"   ├─ Input:      {record.input_summary}")
            print(f"   ├─ Output:     {record.output_summary}")
            print(
                f"   └─ Metrics:    {record.duration_ms:.2f} ms | Confidence: {record.confidence*100:.0f}%"
            )

        print("\n" + "-" * 70)
        print(
            f"⏱️ TOTAL PIPELINE EXECUTION TIME: {state.total_pipeline_time_ms:.2f} ms"
        )
        print("=" * 70)