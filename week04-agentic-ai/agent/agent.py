import json
from typing import Any, Dict
# pyrefly: ignore [missing-import]
from agent.executor import PlanExecutor
# pyrefly: ignore [missing-import]
from agent.memory import ConversationMemory
# pyrefly: ignore [missing-import]
from agent.planner import ReasoningPlanner
# pyrefly: ignore [missing-import]
from agent.registry import ToolRegistry
# pyrefly: ignore [missing-import]
from agent.state import AgentState


class ReasoningAgent:

    def __init__(self, llm_client, registry: ToolRegistry):
        self.llm = llm_client
        self.registry = registry
        self.memory = ConversationMemory()
        self.planner = ReasoningPlanner(llm_client, registry)
        self.executor = PlanExecutor(registry)

    def run(self, user_query: str) -> str:
        print(f"\n{'='*70}\n👤 User Query: {user_query}\n{'='*70}")

        # 1. Initialize State with Conversation Context
        conv_context = self.memory.get_context_string()
        state = AgentState(
            user_query=user_query, conversation_context=conv_context
        )

        # 2. ReAct Iterative Reasoning Loop
        while not state.is_complete and state.step_count < state.max_steps:
            state.step_count += 1
            print(
                f"\n🔄 --- [Step {state.step_count}/{state.max_steps}] Reasoning Cycle ---"
            )

            # A. Plan next action or decide to finish
            decision = self.planner.plan_next_step(state)
            thought = decision.get("thought", "Analyzing query...")
            action = decision.get("action", "TOOL_CALL")

            print(f"  🧠 [Thought]: {thought}")
            state.add_scratchpad_note(thought)

            # B. Check Reflection Condition
            if action == "FINAL_ANSWER":
                print("  💡 [Reflection]: Information sufficient. Ending tool loop.")
                state.is_complete = True
                break

            # C. Execute Tool Action
            tool_name = decision.get("tool")
            raw_args = decision.get("arguments", {})
            if tool_name:
                state = self.executor.execute_step(tool_name, raw_args, state)
            else:
                print("  ⚠️ No tool specified for TOOL_CALL. Completing loop.")
                state.is_complete = True

        # 3. Response Synthesis
        print(
            "\n💬 [Response Synthesis] Generating answer from AgentState scratchpad & history..."
        )
        synthesis_prompt = f"""You are a Legal AI Assistant. Synthesize a complete and accurate answer for the user based on the reasoning scratchpad and execution history.

Conversation History:
{state.conversation_context}

User Query: {state.user_query}

Scratchpad & Execution Observations:
{state.get_scratchpad_formatted()}

Execution History:
{json.dumps(state.history, indent=2)}

Final Answer:"""

        final_response = self.llm.invoke(synthesis_prompt)
        state.final_answer = final_response

        # 4. Save Turn to Conversation Memory
        self.memory.add_turn(user_query, final_response)

        return final_response