"""
agent/agent.py
Master LegalAgent orchestrator coordinating planning, tool execution, and memory.
"""

from typing import Dict, Any
# pyrefly: ignore [missing-import]
from agent.memory import AgentMemory
# pyrefly: ignore [missing-import]
from agent.planner import RuleBasedPlanner
# pyrefly: ignore [missing-import]
from agent.registry import ToolRegistry
# pyrefly: ignore [missing-import]
from agent.executor import PlanExecutor


class LegalAgent:
    def __init__(self):
        self.memory = AgentMemory()
        self.planner = RuleBasedPlanner()
        self.registry = ToolRegistry()
        self.executor = PlanExecutor(self.registry, self.memory)

    def run(self, query: str) -> Dict[str, Any]:
        """Runs the query through Planning -> Execution -> Observation -> Final Answer."""
        self.memory.clear()
        self.memory.add_user_message(query)

        # 1. Generate Execution Plan
        plan = self.planner.create_plan(query)

        # 2. Execute Plan & Record Observations
        execution_result = self.executor.execute_plan(plan)

        # 3. Format Final Answer
        if isinstance(execution_result, list) and execution_result:
            final_answer = execution_result[0].get("document", str(execution_result))
        else:
            final_answer = str(execution_result)

        self.memory.add_final_answer(final_answer)

        return {
            "query": query,
            "plan": plan,
            "final_answer": final_answer,
            "history": self.memory.get_history()
        }