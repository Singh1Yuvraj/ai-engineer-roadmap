import inspect
from typing import Any, Dict, List


class PlanExecutor:

    def __init__(self, tool_registry: Dict[str, Any]):
        self.registry = tool_registry

    def execute(self, plan: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        observations = []

        for step in plan:
            tool_name = step.get("tool")
            tool_input = step.get("input", {})

            print(
                f"  ⚙️ [Executor] Executing tool '{tool_name}' with args: {tool_input}"
            )

            if tool_name not in self.registry:
                obs = f"Error: Tool '{tool_name}' is not registered."
                observations.append({"tool": tool_name, "observation": obs})
                continue

            tool = self.registry[tool_name]

            try:
                # Handle Tool classes vs functions dynamically
                tool_callable = getattr(tool, "run", tool)

                if isinstance(tool_input, dict):
                    # Inspect tool target to safely map arguments
                    sig = inspect.signature(tool_callable)
                    params = sig.parameters

                    # If tool expects 1 positional arg (e.g. string query) but input is dict
                    if len(params) == 1 and not any(
                        p.kind == inspect.Parameter.VAR_KEYWORD
                        for p in params.values()
                    ):
                        single_val = next(iter(tool_input.values()))
                        observation = tool_callable(single_val)
                    else:
                        observation = tool_callable(**tool_input)
                else:
                    observation = tool_callable(tool_input)

            except Exception as e:
                observation = f"Execution Error: {str(e)}"

            print(f"  👁️ [Observation]: {observation}")
            observations.append(
                {"tool": tool_name, "observation": observation}
            )

        return observations