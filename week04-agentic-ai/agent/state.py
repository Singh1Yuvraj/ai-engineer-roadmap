from typing import Any, Dict, List, Optional


class AgentState:

    def __init__(self, user_query: str):
        self.user_query: str = user_query
        self.tool_results: Dict[str, Any] = {}
        self.history: List[Dict[str, Any]] = []
        self.last_result: Optional[Any] = None
        self.errors: List[str] = []

    def record_step(
        self,
        tool_name: str,
        arguments: Dict[str, Any],
        output: Any,
        status: str,
        error: Optional[str] = None,
    ) -> None:
        step_record = {
            "tool": tool_name,
            "arguments": arguments,
            "output": output,
            "status": status,
            "error": error,
        }
        self.history.append(step_record)
        self.tool_results[tool_name] = output
        self.last_result = output
        if error:
            self.errors.append(f"[{tool_name}]: {error}")

    def resolve_arguments(
        self, raw_arguments: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Resolves state variables like '$last' or '$tool_name.output' into real runtime values."""
        resolved = {}
        for key, val in raw_arguments.items():
            if isinstance(val, str) and val.startswith("$"):
                if val == "$last":
                    resolved[key] = (
                        self.last_result if self.last_result is not None else ""
                    )
                elif val.startswith("$") and ".output" in val:
                    target_tool = val[1:].split(".output")[0]
                    resolved[key] = self.tool_results.get(
                        target_tool, self.last_result
                    )
                else:
                    resolved[key] = val
            else:
                resolved[key] = val
        return resolved