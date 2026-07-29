# pyrefly: ignore [missing-import]
from typing import Any
# pyrefly: ignore [missing-import]
from agent.state import AgentState
# pyrefly: ignore [missing-import]
from agent.supervisor import SupervisorAgent
# pyrefly: ignore [missing-import]
from agent.communication import CommunicationBus
# pyrefly: ignore [missing-import]
from agent.agents.research_agent import ResearchAgent
# pyrefly: ignore [missing-import]
from agent.agents.contract_agent import ContractAgent
# pyrefly: ignore [missing-import]
from agent.agents.risk_agent import RiskAgent
# pyrefly: ignore [missing-import]
from agent.agents.response_agent import ResponseAgent

class MultiAgentSystem:
    def __init__(self, llm_client: Any, tool_registry: Any):
        self.llm = llm_client
        self.registry = tool_registry
        self.supervisor = SupervisorAgent(llm_client)
        
        # Instantiate worker team
        self.workers = {
            "ResearchAgent": ResearchAgent(tool_registry),
            "ContractAgent": ContractAgent(tool_registry),
            "RiskAgent": RiskAgent(tool_registry),
            "ResponseAgent": ResponseAgent(llm_client)
        }

    def run(self, user_query: str) -> str:
        print(f"\n{'='*70}\n👤 User Query: {user_query}\n{'='*70}")

        # 1. Initialize Shared AgentState
        state = AgentState(user_query=user_query)

        # 2. Supervisor Orchestrates Execution
        state = self.supervisor.orchestrate(state, self.workers)

        # 3. Print Inter-Agent Trace
        CommunicationBus.print_trace(state)

        return state.final_answer