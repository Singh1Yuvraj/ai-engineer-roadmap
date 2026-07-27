from typing import Any, Dict
# pyrefly: ignore [missing-import]
from agent.registry import ToolRegistry
# pyrefly: ignore [missing-import]
from agent.state import AgentState


class PlanExecutor:

    def __init__(self, registry: ToolRegistry):
        self.registry = registry

    def execute_step(
        self,
        tool_name: str,
        raw_args: Dict[str, Any],
        state: AgentState,
    ) -> AgentState:
        # 1. Resolve argument references ($last -> real output)
        resolved_args = state.resolve_arguments(raw_args)

        print(f"  ⚙️ [Executor] Target Tool: '{tool_name}'")
        print(f"     Raw Args:      {raw_args}")
        print(f"     Resolved Args: {resolved_args}")

        # 2. Validate against tool schema
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
            return state

        # 3. Execute tool
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