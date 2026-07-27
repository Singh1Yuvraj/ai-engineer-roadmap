from typing import Any, Dict, List
# pyrefly: ignore [missing-import]
from agent.registry import ToolRegistry
# pyrefly: ignore [missing-import]
from agent.state import AgentState


class PlanExecutor:

    def __init__(self, registry: ToolRegistry):
        self.registry = registry

    def execute(
        self, tool_calls: List[Dict[str, Any]], state: AgentState
    ) -> AgentState:
        for call in tool_calls:
            tool_name = call.get("tool")
            raw_args = call.get("arguments", {})

            # 1. Resolve arguments using central AgentState (e.g. $last -> actual text)
            resolved_args = state.resolve_arguments(raw_args)

            print(f"\n  ⚙️ [Executor] Target Tool: '{tool_name}'")
            print(f"     Raw Args:      {raw_args}")
            print(f"     Resolved Args: {resolved_args}")

            # 2. Schema Validation
            validation_errors = self.registry.validate_call(
                tool_name, resolved_args
            )
            if validation_errors:
                err_msg = "; ".join(validation_errors)
                print(f"  ❌ Validation Error: {err_msg}")
                state.record_step(
                    tool_name=tool_name,
                    arguments=resolved_args,
                    output=None,
                    status="failed",
                    error=err_msg,
                )
                continue

            # 3. Execution & State Recording
            tool = self.registry.get_tool(tool_name)
            try:
                tool_callable = getattr(tool, "run", tool)
                observation = tool_callable(**resolved_args)
                print(f"  👁️ [Observation]: {observation}")

                state.record_step(
                    tool_name=tool_name,
                    arguments=resolved_args,
                    output=observation,
                    status="completed",
                )
            except Exception as e:
                err_msg = f"Execution Exception: {str(e)}"
                print(f"  ❌ Execution Error: {err_msg}")
                state.record_step(
                    tool_name=tool_name,
                    arguments=resolved_args,
                    output=None,
                    status="failed",
                    error=err_msg,
                )

        return state