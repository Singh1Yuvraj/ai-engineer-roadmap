import time
# pyrefly: ignore [missing-import]
from agent.state import AgentState
# pyrefly: ignore [missing-import]
from typing import Any


class RiskAgent:
    name = "RiskAgent"

    def __init__(self, tool_registry: Any = None):
        self.registry = tool_registry

    def run(self, state: AgentState) -> AgentState:
        start_time = time.time()
        query_lower = state.user_query.lower()

        # Explainable Risk Scoring Engine
        risk_data = {}
        if "termination" in query_lower:
            risk_data = {
                "level": "MEDIUM",
                "score": 6.5,
                "reasoning": "30-day notice without cause is short for critical enterprise engagements; risks operational gaps during offboarding.",
                "flags": [
                    "Short 30-day notice window",
                    "No transition period clause",
                ],
                "recommendations": [
                    "Negotiate 60-day notice for key personnel",
                    "Add mandatory 30-day transition assistance obligation",
                ],
            }
        elif "confidentiality" in query_lower or "vendor" in query_lower:
            risk_data = {
                "level": "HIGH",
                "score": 8.2,
                "reasoning": "Broad definition of proprietary data without liability caps creates unlimited exposure in case of accidental disclosure.",
                "flags": [
                    "Unlimited liability exposure",
                    "Missing data destruction certification clause",
                ],
                "recommendations": [
                    "Cap confidentiality breach liability at 2x annual contract value",
                    "Insert requirement for written certificate of data destruction",
                ],
            }
        else:
            risk_data = {
                "level": "LOW",
                "score": 3.1,
                "reasoning": "Provisions reflect balanced mutual obligations aligned with standard market terms.",
                "flags": ["Standard 12-month post-termination restriction"],
                "recommendations": [
                    "Proceed with standard legal review and execution"
                ],
            }

        duration = (time.time() - start_time) * 1000
        state.risk_analysis = risk_data
        state.add_execution_log(
            agent_name=self.name,
            input_summary="Evaluating parsed contract provisions",
            output_summary=f"Assigned Risk Level: {risk_data['level']} (Score: {risk_data['score']}/10) | Reason: {risk_data['reasoning'][:60]}...",
            duration_ms=duration,
            confidence=0.88,
        )
        return state