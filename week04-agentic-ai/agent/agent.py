import json
from typing import Any, Dict
# pyrefly: ignore [missing-import]
from agent.executor import PlanExecutor
# pyrefly: ignore [missing-import]
from agent.planner import FunctionPlanner
# pyrefly: ignore [missing-import]
from agent.registry import ToolRegistry
# pyrefly: ignore [missing-import]
from agent.state import AgentState


class FunctionAgent:

    def __init__(self, llm_client, registry: ToolRegistry):
        self.llm = llm_client
        self.registry = registry
        self.planner = FunctionPlanner(llm_client, registry)
        self.executor = PlanExecutor(registry)

    def run(self, user_query: str) -> str:
        print(f"\n{'='*70}\n👤 User Query: {user_query}\n{'='*70}")

        # 1. Initialize Central AgentState
        state = AgentState(user_query=user_query)

        # 2. Generate tool calls with $last state bindings
        print("🧠 [1. Planner] Evaluating query against tool schemas...")
        tool_calls = self.planner.create_plan(user_query)
        print(
            f"📋 [2. Tool Call Plan Generated]: {len(tool_calls)} action(s)"
        )

        # 3. Pass BOTH tool_calls AND state to executor
        print("\n⚡ [3. Executor] Running tool call sequence...")
        state = self.executor.execute(tool_calls, state)

        # 4. Synthesize response using state execution history
        print(
            "\n💬 [4. Final LLM] Synthesizing response from AgentState execution history..."
        )
        synthesis_prompt = f"""
Synthesize a concise and accurate response for the user using ONLY the tool execution history in AgentState.

User Query: {state.user_query}
Execution History: {json.dumps(state.history, indent=2)}

Final Response:
"""
        final_answer = self.llm.invoke(synthesis_prompt)
        return final_answer