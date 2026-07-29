import time
# pyrefly: ignore [missing-import]
from typing import Any
# pyrefly: ignore [missing-import]
from agent.state import AgentState


class ResponseAgent:
    name = "ResponseAgent"

    def __init__(self, llm_client: Any = None):
        self.llm = llm_client

    def run(self, state: AgentState) -> AgentState:
        start_time = time.time()

        retrieved = state.retrieved_documents
        contract = state.contract_analysis
        risk = state.risk_analysis

        # Calculate overall confidence
        avg_confidence = (
            sum(r.confidence for r in state.execution_records)
            / len(state.execution_records)
            if state.execution_records
            else 0.90
        )

        report = [
            f"=========================================================",
            f"          LEGAL INTELLIGENCE EXECUTIVE REPORT            ",
            f"=========================================================",
            f"📌 USER QUERY: {state.user_query}",
            f"📊 OVERALL CONFIDENCE SCORE: {avg_confidence*100:.1f}%\n",
            f"---------------------------------------------------------",
            f"1. EXECUTIVE SUMMARY",
            f"---------------------------------------------------------",
            f"{contract.get('summary', 'Analysis completed across retrieved contract provisions.')}\n",
            f"---------------------------------------------------------",
            f"2. KEY LEGAL FINDINGS & EVIDENCE",
            f"---------------------------------------------------------",
        ]

        for finding in contract.get("findings", []):
            report.append(f"• {finding}")

        report.extend(
            [
                f"\n📌 CITED SOURCES:",
                f"  " + "\n  ".join(contract.get("citations", ["None"])),
            ]
        )

        if risk:
            report.extend(
                [
                    f"\n---------------------------------------------------------",
                    f"3. RISK ASSESSMENT & EXPLAINABLE REASONING",
                    f"---------------------------------------------------------",
                    f"• SEVERITY LEVEL: [{risk.get('level', 'N/A')}] (Score: {risk.get('score', 'N/A')}/10)",
                    f"• REASONING: {risk.get('reasoning', 'N/A')}",
                    f"• RISK FLAGS:",
                    f"  - " + "\n  - ".join(risk.get("flags", [])),
                    f"\n---------------------------------------------------------",
                    f"4. NEGOTIATION SUGGESTIONS & MITIGATION",
                    f"---------------------------------------------------------",
                    f"  - "
                    + "\n  - ".join(risk.get("recommendations", [])),
                ]
            )

        duration = (time.time() - start_time) * 1000
        state.final_answer = "\n".join(report)
        state.add_execution_log(
            agent_name=self.name,
            input_summary="Synthesizing multi-agent data contracts",
            output_summary=f"Generated executive report with citations and recommendations.",
            duration_ms=duration,
            confidence=0.95,
        )
        return state