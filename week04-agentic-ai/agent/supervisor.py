import time
from typing import Any, List
# pyrefly: ignore [missing-import]
from agent.state import AgentState


class SupervisorAgent:
    name = "SupervisorAgent"

    def __init__(self, llm_client: Any = None):
        self.llm = llm_client

    def plan_workflow(self, query: str) -> List[str]:
        q = query.lower()
        if "recommendation" in q or "vendor" in q or "stronger" in q:
            return [
                "ResearchAgent",
                "ContractAgent",
                "RiskAgent",
                "ResponseAgent",
            ]
        elif "risk" in q or "termination" in q:
            return ["ResearchAgent", "RiskAgent", "ResponseAgent"]
        elif "confidentiality" in q or "summarize" in q:
            return ["ResearchAgent", "ContractAgent", "ResponseAgent"]
        else:
            return [
                "ResearchAgent",
                "ContractAgent",
                "RiskAgent",
                "ResponseAgent",
            ]

    def orchestrate(self, state: AgentState, workers: dict) -> AgentState:
        total_start = time.time()
        workflow = self.plan_workflow(state.user_query)

        print(f"\n  🧠 [Supervisor] Query Intent Analyzed.")
        print(f"  📋 [Workflow Plan]: {' ➔ '.join(workflow)}")

        for agent_name in workflow:
            if agent_name in workers:
                print(f"\n  🔀 [Handoff] Supervisor ➔ {agent_name}")
                worker = workers[agent_name]
                state = worker.run(state)

        state.total_pipeline_time_ms = (time.time() - total_start) * 1000
        return state