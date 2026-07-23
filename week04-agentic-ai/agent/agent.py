from typing import Any, Dict
# pyrefly: ignore [missing-import]
from agent.executor import PlanExecutor
# pyrefly: ignore [missing-import]
from agent.planner import LLMPlanner


class LLMPlanningAgent:

    def __init__(self, llm_client, tool_registry: Dict[str, Any]):
        self.llm = llm_client
        self.planner = LLMPlanner(llm_client)
        self.executor = PlanExecutor(tool_registry)

    def run(self, user_query: str) -> str:
        print(f"\n{'='*70}\n👤 User Query: {user_query}\n{'='*70}")

        # Step 1: LLM Planning Phase
        print("🧠 [1. LLM Planner] Requesting execution plan...")
        plan = self.planner.create_plan(user_query)
        print(f"📋 [2. Parsed JSON Plan]: {plan}")

        # Step 2: Deterministic Execution Phase (Zero LLM involvement here)
        print("⚡ [3. Executor] Running plan against tool registry...")
        observations = self.executor.execute(plan)

        # Step 3: Synthesis Phase (LLM summarizes observations into final user answer)
        print("💬 [4. Final Answer LLM] Synthesizing response from observations...")
        synthesis_prompt = f"""
You are a Legal AI assistant. Answer the user's question accurately using ONLY the tool observations provided below.

User Query: {user_query}
Tool Observations: {observations}

Final Answer:
"""
        final_answer = self.llm.invoke(synthesis_prompt)
        return final_answer